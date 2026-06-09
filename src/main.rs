mod layers;
mod musetalk;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use hanzo_ml::{DType, Device, Result, Shape, Tensor};
use hanzo_nn::var_builder::SimpleBackend;
use hanzo_nn::Init;
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

use musetalk::{MuseTalk, MuseTalkConfig};

fn weight_std() -> f32 {
    std::env::var("MUSETALK_WSTD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02)
}

struct SeededBackend {
    seed: AtomicU64,
}

impl SeededBackend {
    fn new(seed: u64) -> Self {
        Self {
            seed: AtomicU64::new(seed),
        }
    }
    fn next(&self) -> f32 {
        let mut x = self.seed.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed.store(x, Ordering::Relaxed);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

impl SimpleBackend for SeededBackend {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let n: usize = s.elem_count();
        if name.ends_with("bias") {
            return Tensor::zeros(s, dtype, dev);
        }
        if name.ends_with("weight") && s.rank() == 1 {
            return Tensor::ones(s, dtype, dev);
        }
        let wstd = weight_std();
        let data: Vec<f32> = (0..n).map(|_| self.next() * wstd).collect();
        Tensor::from_vec(data, s, &Device::Cpu)?
            .to_device(dev)?
            .to_dtype(dtype)
    }
    fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        hanzo_ml::bail!("SeededBackend requires an explicit shape")
    }
    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn seeded_vb(seed: u64, dtype: DType, dev: &Device) -> ShardedVarBuilder {
    ShardedSafeTensors::wrap(Box::new(SeededBackend::new(seed)), dtype, dev.clone())
}

fn seeded_input(seed: u64, shape: &[usize], dev: &Device) -> Result<Tensor> {
    let b = SeededBackend::new(seed);
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (b.next() + 1.0) * 0.5).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(dev)
}

fn psnr_cosine(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len());
    let mut mse = 0f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        mse += (x - y) * (x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    mse /= a.len() as f64;
    let psnr = if mse <= 1e-12 { 120.0 } else { 10.0 * (1.0 / mse).log10() };
    let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    Ok((psnr, cosine))
}

struct Stage {
    encode: f64,
    unet: f64,
    decode: f64,
}

fn time_frame(model: &MuseTalk, face: &Tensor, audio: &Tensor, dev: &Device) -> Result<Stage> {
    dev.synchronize()?;
    let t0 = Instant::now();
    let latents = model.latents_for_unet(face)?;
    dev.synchronize()?;
    let t1 = Instant::now();
    let b = latents.dim(0)?;
    let ts = Tensor::zeros(b, DType::F32, dev)?;
    let pred = model.unet_forward(&latents, &ts, audio)?;
    dev.synchronize()?;
    let t2 = Instant::now();
    let _img = model.decode_latents(&pred)?;
    dev.synchronize()?;
    let t3 = Instant::now();
    Ok(Stage {
        encode: (t1 - t0).as_secs_f64() * 1e3,
        unet: (t2 - t1).as_secs_f64() * 1e3,
        decode: (t3 - t2).as_secs_f64() * 1e3,
    })
}

fn pick_dtype() -> DType {
    match std::env::var("MUSETALK_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    }
}

fn run_bench(dev: &Device) -> Result<()> {
    let dtype = pick_dtype();
    let iters: usize = std::env::var("MUSETALK_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let cfg = MuseTalkConfig::default();
    let model = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, dev),
        seeded_vb(0x9abc_def0, dtype, dev),
        dev,
        dtype,
    )?;
    let bsz: usize = std::env::var("MUSETALK_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sz = cfg.resized_img;
    let face = seeded_input(0x55, &[bsz, 3, sz, sz], dev)?;
    let audio = seeded_input(0xAA, &[bsz, 50, cfg.unet.cross_attention_dim], dev)?.to_dtype(dtype)?;

    for _ in 0..3 {
        let _ = time_frame(&model, &face, &audio, dev)?;
    }

    let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
    let mut total_min = f64::MAX;
    for _ in 0..iters {
        let s = time_frame(&model, &face, &audio, dev)?;
        e += s.encode;
        u += s.unet;
        d += s.decode;
        total_min = total_min.min(s.encode + s.unet + s.decode);
    }
    let n = iters as f64;
    let (e, u, d) = (e / n / bsz as f64, u / n / bsz as f64, d / n / bsz as f64);
    let total = e + u + d;
    let total_min = total_min / bsz as f64;
    println!("\n==== MuseTalk bench  dev={:?} dtype={:?} iters={} batch={} (per-frame) ====", dev.location(), dtype, iters, bsz);
    println!("vae-encode(x2): {:8.3} ms", e);
    println!("unet-1step:     {:8.3} ms", u);
    println!("vae-decode:     {:8.3} ms", d);
    println!("total/frame:    {:8.3} ms  (best {:.3} ms)", total, total_min);
    println!("fps(mean):      {:8.2}", 1000.0 / total);
    println!("fps(best):      {:8.2}", 1000.0 / total_min);
    Ok(())
}

fn run_verify() -> Result<()> {
    let cpu = Device::Cpu;
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img;

    let model_cpu = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, DType::F32, &cpu),
        seeded_vb(0x9abc_def0, DType::F32, &cpu),
        &cpu,
        DType::F32,
    )?;
    let face_cpu = seeded_input(0x55, &[1, 3, sz, sz], &cpu)?;
    let audio_cpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &cpu)?;
    let ref_img = model_cpu.forward(&face_cpu, &audio_cpu)?;

    let gpu = Device::new_cuda(0)?;
    let dtype = pick_dtype();
    let model_gpu = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, &gpu),
        seeded_vb(0x9abc_def0, dtype, &gpu),
        &gpu,
        dtype,
    )?;
    let face_gpu = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?;
    let audio_gpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?.to_dtype(dtype)?;
    let gpu_img = model_gpu.forward(&face_gpu, &audio_gpu)?;

    let lat_cpu = model_cpu.latents_for_unet(&face_cpu)?;
    let lat_gpu = model_gpu.latents_for_unet(&face_gpu)?;
    let (lp, lc) = psnr_cosine(&lat_cpu, &lat_gpu.to_device(&cpu)?)?;

    let ts_cpu = Tensor::zeros(1, DType::F32, &cpu)?;
    let ts_gpu = Tensor::zeros(1, DType::F32, &gpu)?;
    let lat_gpu_id = lat_cpu.to_device(&gpu)?.to_dtype(dtype)?;
    let pred_cpu = model_cpu.unet_forward(&lat_cpu, &ts_cpu, &audio_cpu)?;
    let pred_gpu = model_gpu.unet_forward(&lat_gpu_id, &ts_gpu, &audio_gpu)?;
    let (up, uc) = psnr_cosine(&pred_cpu, &pred_gpu.to_device(&cpu)?)?;

    let pred_gpu_id = pred_cpu.to_device(&gpu)?.to_dtype(dtype)?;
    let dec_cpu = model_cpu.decode_latents(&pred_cpu)?;
    let dec_gpu = model_gpu.decode_latents(&pred_gpu_id)?;
    let (dp, dc) = psnr_cosine(&dec_cpu, &dec_gpu.to_device(&cpu)?)?;

    let (psnr, cosine) = psnr_cosine(&ref_img, &gpu_img.to_device(&cpu)?)?;
    println!("\n==== MuseTalk correctness  gpu_dtype={:?} vs cpu-f32 ====", dtype);
    println!("stage latents_for_unet: PSNR {:7.3} dB  cosine {:.6}", lp, lc);
    println!("stage unet (on same lat): PSNR {:7.3} dB  cosine {:.6}", up, uc);
    println!("stage decode (on same pred): PSNR {:7.3} dB  cosine {:.6}", dp, dc);
    println!("full forward:           PSNR {:7.3} dB  cosine {:.6}", psnr, cosine);
    Ok(())
}

// GPU-vs-GPU same-dtype fidelity gate: save the per-stage GPU outputs as the pre-lever
// reference (MUSETALK_REF_SAVE=1), then after a lever re-run and compare. Any divergence is
// purely from the kernel change, not from the f16-vs-f32 cross-device gap that `verify` mixes in.
fn run_selfcheck() -> Result<()> {
    let dir = std::env::var("MUSETALK_REF_DIR").unwrap_or_else(|_| "/tmp/musetalk_ref".to_string());
    let save = std::env::var("MUSETALK_REF_SAVE").is_ok();
    let gpu = Device::new_cuda(0)?;
    let dtype = pick_dtype();
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img;
    let model = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, &gpu),
        seeded_vb(0x9abc_def0, dtype, &gpu),
        &gpu,
        dtype,
    )?;
    let face = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?;
    let audio = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?.to_dtype(dtype)?;

    let lat = model.latents_for_unet(&face)?;
    let ts = Tensor::zeros(1, DType::F32, &gpu)?;
    let pred = model.unet_forward(&lat, &ts, &audio)?;
    let dec = model.decode_latents(&pred)?;
    let stages = [("encode", &lat), ("unet", &pred), ("decode", &dec)];

    if save {
        std::fs::create_dir_all(&dir).ok();
        for (name, t) in stages.iter() {
            t.to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .write_npy(format!("{dir}/{name}.npy"))?;
        }
        println!("saved GPU reference (dtype={dtype:?}) to {dir}");
    } else {
        println!("\n==== MuseTalk self-check  gpu_dtype={dtype:?} vs saved GPU ref ====");
        let mut worst = f64::MAX;
        for (name, t) in stages.iter() {
            let r = Tensor::read_npy(format!("{dir}/{name}.npy"))?.to_device(&Device::Cpu)?;
            let (p, c) = psnr_cosine(&r, &t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)?;
            worst = worst.min(p);
            println!("stage {name:7}: PSNR {p:8.3} dB  cosine {c:.6}");
        }
        println!("worst-stage PSNR: {worst:.3} dB  (>=60 dB = numerically faithful)");
    }
    Ok(())
}

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "bench".to_string());
    let dev = match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    match mode.as_str() {
        "verify" => run_verify()?,
        "selfcheck" => run_selfcheck()?,
        _ => run_bench(&dev)?,
    }
    Ok(())
}
