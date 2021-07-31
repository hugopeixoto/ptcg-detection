use detection::*;
use image::GenericImage;
use std::time::Instant;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

fn main() {
    let dataset = load_or_build_dataset("dataset/", "dataset.txt");

    let width = 1920;
    let height = 1080;

    let mut buffers = ProcessingBuffers::new(width, height);

    let mut dev = v4l::Device::new(0).expect("Failed to open device");
    println!("{:?}", dev.format());

    for format in dev.enum_formats().unwrap() {
        println!("  {} ({})", format.fourcc, format.description);
    }

    let mut stream = v4l::io::mmap::Stream::with_buffers(&mut dev, v4l::buffer::Type::VideoCapture, 4)
                .expect("Failed to create buffer stream");

    loop {
        let mut t = Instant::now();
        let (frame, meta) = stream.next().unwrap();

        let mut processing = ProcessingPipeline { frame: &frame, buffers: &mut buffers };

        // I shouldn't care about RGB. Luma is all I need to calculate the img hash
        for y in 0..height as usize {
            let yif = 2 * width as usize * y;
            for x in (0..width as usize).step_by(2) {
                let y1 = frame[yif + 2 * x + 0];
                let u  = frame[yif + 2 * x + 1];
                let y2 = frame[yif + 2 * x + 2];
                let v  = frame[yif + 2 * x + 3];

                let c1: i32 = y1 as i32 - 16;
                let c2: i32 = y2 as i32 - 16;
                let d: i32  = u  as i32 - 128;
                let e: i32  = v  as i32 - 128;

                let r1: u8 = ((298 * c1 + 409 * e + 128) >> 8).clamp(0, 255) as u8;
                let g1: u8 = ((298 * c1 - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
                let b1: u8 = ((298 * c1 + 516 * d + 128) >> 8).clamp(0, 255) as u8;

                let r2: u8 = ((298 * c2 + 409 * e + 128) >> 8).clamp(0, 255) as u8;
                let g2: u8 = ((298 * c2 - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
                let b2: u8 = ((298 * c2 + 516 * d + 128) >> 8).clamp(0, 255) as u8;

                processing.buffers.source_image.put_pixel(x as u32, y as u32, image::Rgba([r1, g1, b1, 255]));
                processing.buffers.source_image.put_pixel(x as u32 + 1, y as u32, image::Rgba([r2, g2, b2, 255]));
            }
        }
        //println!("[{:?}] YUYV2RGB", t.elapsed());
        //t = Instant::now();

        let (times, best) = process(&mut processing, &dataset);
        // println!("[{:?}] process", t.elapsed());

        match best {
            Some((entry, score, second, score2)) => {
                println!("matches: {:?} ({}) | {:?} ({})", entry.path, score, second.path, score2);
            }
            None => {}
        }
    }
}
