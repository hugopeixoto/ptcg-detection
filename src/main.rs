use detection::{DatasetEntry, load_or_build_dataset};

use image::GenericImage;
use image::GenericImageView;
use std::time::Instant;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

fn process(processing: &mut ProcessingPipeline, dataset: &Vec<DatasetEntry>, stem: &str, debug_images: bool) -> ProcessingTimes {
    let mut times = ProcessingTimes::default();

    if debug_images { processing.buffers.source_image.save(format!("outputs/{}.original.png", stem)).unwrap(); }

    let mut time = Instant::now();

    let width = processing.buffers.width;
    let height = processing.buffers.height;
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();

    processing.calculate_sobel();
    times.sobel = time.elapsed();
    time = Instant::now();

    processing.calculate_border();
    times.border = time.elapsed();

    let hough = calculate_hough(&processing.buffers.border, width, height);
    times.hough = time.elapsed();
    time = Instant::now();

    let angles = 900;
    let rhos = 900;

    let mut lines = vec![];
    for a in 0..angles {
        for r in 0..rhos {
            let x = hough[a * rhos + r];
            if x > 200 {
                lines.push((a, r));
            }
        }
    }

    // eprintln!(":: found lines ({})", lines.len());

    let mut clustered_lines: Vec<(f64, f64, usize)> = vec![];
    for point in lines.iter() {
        let dup = clustered_lines.iter().position(|p| ((p.0/p.2 as f64) - point.0 as f64).abs() < 20.0 && ((p.1/p.2 as f64) - point.1 as f64).abs() < 50.0);
        match dup {
            None => {
                clustered_lines.push((point.0 as f64, point.1 as f64, 1));
            },
            Some(index) => {
                let (p, r, c) = clustered_lines[index];
                clustered_lines[index] = (p + point.0 as f64, r + point.1 as f64, c + 1);
            }
        }
    }

    // eprintln!(":: clustered lines ({})", clustered_lines.len());

    let mut intersections = vec![];
    for i1 in 0..clustered_lines.len() {
        let p1 = clustered_lines[i1];
        for i2 in i1 + 1 .. clustered_lines.len() {
            let p2 = clustered_lines[i2];
            // find intersection
            let a1 = p1.0 / (p1.2 as f64) * 2.0 * std::f64::consts::PI / angles as f64;
            let a2 = p2.0 / (p2.2 as f64) * 2.0 * std::f64::consts::PI / angles as f64;
            let r1 = p1.1 / p1.2 as f64;
            let r2 = p2.1 / p2.2 as f64;
            let ct1 = a1.cos();
            let ct2 = a2.cos();
            let st1 = a1.sin();
            let st2 = a2.sin();
            let det = ct1 * st2 - st1 * ct2;
            if det != 0.0 {
                let x3 = (st2 * (r1 * diagonal / rhos as f64) - st1 * (r2 * diagonal / rhos as f64)) / det;
                let y3 = (-ct2 * (r1 * diagonal / rhos as f64) + ct1 * (r2 * diagonal / rhos as f64)) / det;

                if 0.0 <= x3 && x3 < width as f64 && 0.0 <= y3 && y3 < height as f64 {
                    intersections.push((x3 as u32, y3 as u32));
                }
            }
        }
    }

    // eprintln!(":: found intersections ({})", intersections.len());

    let mut clustered_intersections: Vec<(f64, f64)> = vec![];
    let mut counts = vec![];
    for point in intersections.iter() {
        let dup = clustered_intersections.iter().position(|p| ((p.0 - point.0 as f64).powi(2) + (p.1 - point.1 as f64).powi(2)) < 256.0);
        match dup {
            None => {
                clustered_intersections.push((point.0 as f64, point.1 as f64));
                counts.push(1);
            },
            Some(index) => {
                let (p, r) = clustered_intersections[index];
                clustered_intersections[index] = (p + point.0 as f64, r + point.1 as f64);
                counts[index] += 1;
            }
        }
    }

    for i in 0..counts.len() {
        clustered_intersections[i].0 = clustered_intersections[i].0 / counts[i] as f64;
        clustered_intersections[i].1 = clustered_intersections[i].1 / counts[i] as f64;
    }

    // eprintln!(":: clustered intersections ({})", clustered_intersections.len());

    let mut redundant = vec![];
    redundant.resize(clustered_intersections.len(), false);
    for i in 0..clustered_intersections.len() {
        let p1 = clustered_intersections[i];
        for j in i+1..clustered_intersections.len() {
            let p2 = clustered_intersections[j];
            for k in j+1..clustered_intersections.len() {
                let p3 = clustered_intersections[k];

                let area = 0.5 * (p1.0 * (p2.1 - p3.1) + p2.0 * (p3.1 - p1.1) + p3.0 * (p1.1 - p2.1)).abs();
                if area < 200.0 {
                    let d12 = (p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2);
                    let d23 = (p2.0 - p3.0).powi(2) + (p2.1 - p3.1).powi(2);
                    let d13 = (p1.0 - p3.0).powi(2) + (p1.1 - p3.1).powi(2);

                    if d12 >= d23 && d12 >= d13 { redundant[k] = true; }
                    if d23 >= d13 && d23 >= d12 { redundant[i] = true; }
                    if d13 >= d12 && d13 >= d23 { redundant[j] = true; }
                }
            }
        }
    }

    let mut final_intersections = vec![];
    for i in 0..clustered_intersections.len() {
        if !redundant[i] {
            final_intersections.push(clustered_intersections[i]);
        }
    }

    // eprintln!(":: removed parallel intersections ({:?})", final_intersections);

    if debug_images {
        let mut border_image = image::DynamicImage::new_rgba8(width, height);
        for y in 0..height as usize {
            for x in 0..width as usize {
                if processing.buffers.border[y * width as usize + x] > 0 {
                    border_image.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 0, 255]));
                }
            }
        }
        border_image.save(format!("outputs/{}.border.png", stem)).unwrap();
    }

    if debug_images {
        let mut lines_image = processing.buffers.source_image.clone();
        for &(a, r_h) in lines.iter() {
            for d_abs in 0..=(20*diagonal as usize) {
                let d = (d_abs as f64 / 10.0) - diagonal;
                let r = r_h as f64 * diagonal / rhos as f64;
                let r2 = (r*r + d*d).sqrt();
                let d2 = (a as f64 * 2.0 * std::f64::consts::PI / angles as f64) - (d / r2).asin();

                let x = r2 * d2.cos();
                let y = r2 * d2.sin();

                if 0.0 <= x && x < width as f64 && 0.0 <= y && y < height as f64 {
                    lines_image.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 255, 255]));
                }
            }
        }

        for i in final_intersections.iter() {
            for dx in -7 ..= 7 {
                for dy in -7 ..= 7 {
                    let x = i.0 as i32 + dx;
                    let y = i.1 as i32 + dy;
                    if 0 <= x && x < width as i32 && 0 <= y && y < height as i32 {
                        lines_image.put_pixel(
                            x as u32, y as u32,
                            if dx.abs() >= 6 || dy.abs() >= 6 {
                                image::Rgba([0, 0, 0, 255])
                            } else {
                                image::Rgba([255, 255, 0, 255])
                            },
                        );
                    }
                }
            }
        }


        lines_image.save(format!("outputs/{}.hough-lines.png", stem)).unwrap();
    }

    if final_intersections.len() != 4 {
        return times;
    }

    let mut ordered_intersections = vec![];
    for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
        // find closest point to image corner
        ordered_intersections.push(final_intersections.iter().min_by_key(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)) as u32).unwrap());
    }

    // eprintln!(":: sorted intersections ({:?})", ordered_intersections);
    times.corners = time.elapsed();
    time = Instant::now();

    let a3 = nalgebra::Matrix3::new(
        ordered_intersections[0].0, ordered_intersections[1].0, ordered_intersections[2].0,
        ordered_intersections[0].1, ordered_intersections[1].1, ordered_intersections[2].1,
        1.0, 1.0, 1.0,
    );

    let a1 = nalgebra::Vector3::new(ordered_intersections[3].0, ordered_intersections[3].1, 1.0);
    let ax = match a3.lu().solve(&a1) {
        Some(x) => x,
        None => { return times; }
    };
    let a = a3 * nalgebra::Matrix3::new(
        ax[0],   0.0,   0.0,
          0.0, ax[1],   0.0,
          0.0,   0.0, ax[2],
    );

    // 734x1024
    // for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
    let b3 = nalgebra::Matrix3::new(
        0.0, 734.0,    0.0,
        0.0,  0.0,  1024.0,
        1.0,  1.0,     1.0,
    );

    let b1 = nalgebra::Vector3::new(
         734.0,
        1024.0,
           1.0,
    );

    let bx = b3.lu().solve(&b1).unwrap();
    let b = b3 * nalgebra::Matrix3::new(
        bx[0],   0.0,   0.0,
          0.0, bx[1],   0.0,
          0.0,   0.0, bx[2],
    );

    let c = a * b.try_inverse().unwrap();

    let mut perspective_image = image::DynamicImage::new_rgba8(734, 1024).to_rgba8();
    for y in 0..1024 {
        for x in 0..734 {
            let p = c * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
            let px = (p[0] / p[2]) as i32;
            let py = (p[1] / p[2]) as i32;

            if 0 <= px && px < width as i32 && 0 <= py && py < height as i32 {
                let pixel = processing.buffers.source_image.get_pixel(px as u32, py as u32);
                perspective_image.put_pixel(x, y, pixel);
            }
        }
    }

    times.perspective = time.elapsed();
    time = Instant::now();

    let hasher = img_hash::HasherConfig::new().hash_size(16,16).hash_alg(img_hash::HashAlg::Gradient).to_hasher();

    let hash = hasher.hash_image(&perspective_image);
    let best = dataset.iter().min_by_key(|entry| hash.dist(&entry.hash)).unwrap();

    times.phash = time.elapsed();

    println!("match: {:?} ({})", best.path, hash.dist(&best.hash));

    // perspective_image.save(format!("outputs/{}.perspective.png", stem)).unwrap();
    // image::open(&best.path).unwrap().save(format!("outputs/{}.best.png", stem)).unwrap();
    // println!("[{:?}] debug_images", time.elapsed());

    // let colors = colors(&perspective_image);
    // eprintln!("[{:?}] best color: {:?}", time.elapsed(), colors[0]);
    // let mut color_image = image::DynamicImage::new_rgba8(20, 5);
    // for x in 0..5 {
    //     for c in 0..5 {
    //         color_image.put_pixel(x, c as u32, Rgba([colors[c].0 * bucket, colors[c].1*bucket, colors[c].2*bucket, 255]));
    //     }
    //
    //     for i in 0..3 {
    //         for c in 0..5 {
    //             let color = dataset[i].colors[c];
    //             color_image.put_pixel(x + 5 * (i as u32 + 1), c as u32, Rgba([color.0 * bucket, color.1*bucket, color.2*bucket, 255]));
    //         }
    //     }
    // }
    //
    // color_image.save(format!("outputs/{}.color.png", stem)).unwrap();
    times
}

fn main() {
    let dataset = load_or_build_dataset("dataset/", "dataset.txt");

    // if false {
    //     let paths = std::fs::read_dir("images/canon/").unwrap();
    //     for path in paths {
    //         let filename = path.unwrap().path();
    //         let source_image = image::open(&filename).expect("failed to read image");
    //         let downscaled = source_image.resize(800, 800, image::imageops::FilterType::Triangle);

    //         let time = Instant::now();

    //         process(&downscaled, &dataset, filename.file_stem().unwrap().to_str().unwrap(),
    //         false);
    //         let time = Instant::now();

    //         eprintln!("[{:?}] processed {:?}", time.elapsed(), filename);
    //     }
    // }

    if true {
        let mut buffers = ProcessingBuffers::new(1920, 1080);

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
            println!(
                "\n\n::::: Buffer size: {}, seq: {}, timestamp: {}",
                frame.len(),
                meta.sequence,
                meta.timestamp,
                );

            let mut processing = ProcessingPipeline { frame: &frame, buffers: &mut buffers };

            for y in 0..1080 {
                let yif = 2 * 1920 * y;
                for x in (0..1920).step_by(2) {
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
            println!("[{:?}] YUYV2RGB", t.elapsed());
            t = Instant::now();

            let times = process(&mut processing, &dataset, &format!("webcam.{}", meta.sequence), false);
            println!("[{:?}] sobel", times.sobel);
            println!("[{:?}] border", times.border);
            println!("[{:?}] hough", times.hough);
            println!("[{:?}] corners", times.corners);
            println!("[{:?}] perspective", times.perspective);
            println!("[{:?}] phash", times.phash);
            println!("[{:?}] process", t.elapsed());
        }
    }
}
