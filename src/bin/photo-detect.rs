use detection::*;
use std::time::Instant;
use image::GenericImage;

fn main() {
    let dataset = load_or_build_dataset("dataset/", "dataset.txt");
    let templates = load_templates();

    let width = 1920;
    let height = 1080;

    let mut buffers = ProcessingBuffers::new(width, height);
    let mut frame = vec![];
    frame.resize((width * height * 2) as usize, 0);

    let paths = std::fs::read_dir("images/canon-1080p/").unwrap();
    for path in paths {
        let filename = path.unwrap().path();

        buffers.source_image = image::open(&filename).expect("failed to read image").resize(width, height, image::imageops::Nearest);

        // I should be able to pass this luma8 image using traits
        // instead of having to build a custom frame
        let luma8 = buffers.source_image.to_luma8();
        buffers.width = luma8.width();
        buffers.height = luma8.height();

        for y in 0 .. luma8.height() {
            for x in 0 .. luma8.width() {
                frame[(y * luma8.width() + x) as usize * 2] = luma8.get_pixel(x, y)[0];
            }
        }

        let mut processing = ProcessingPipeline {
            frame: &frame,
            buffers: &mut buffers,
        };

        let time = Instant::now();

        println!("processing");
        let (times, best, detected_set) = process(&mut processing, &dataset, &templates);

        println!("detected set: {:?}", detected_set);

        // println!("[{:?}] sobel", times.sobel);
        // println!("[{:?}] border", times.border);
        // println!("[{:?}] hough", times.hough);
        // println!("[{:?}] corners", times.corners);
        // println!("[{:?}] perspective", times.perspective);
        // println!("[{:?}] phash", times.phash);
        // println!("[{:?}] processed {:?}", time.elapsed(), filename);

        let stem = filename.file_stem().unwrap().to_str().unwrap();
        save_debug_images(&processing, stem, best);
    }
}

fn save_debug_images(processing: &ProcessingPipeline, stem: &str, best: Option<Vec<(&DatasetEntry, u32)>>) {
    let width = processing.buffers.width;
    let height = processing.buffers.height;

    let mut sobel_img = image::DynamicImage::new_luma8(width, height).to_luma8();
    for y in 0..height {
        for x in 0..width {
            let pixel = image::Luma([processing.buffers.sobel[(y * width + x) as usize] as u8]);
            sobel_img.put_pixel(x, y, pixel);
        }
    }

    let mut border_img = image::DynamicImage::new_luma8(width, height).to_luma8();
    for y in 0..height {
        for x in 0..width {
            let pixel = image::Luma([processing.buffers.border[(y * width + x) as usize] as u8 * 255]);
            border_img.put_pixel(x, y, pixel);
        }
    }

    let angles = 900;
    let rhos = 900;
    let mut hough_img = image::DynamicImage::new_luma8(angles, rhos).to_luma8();
    for a in 0..angles {
        for r in 0..rhos {
            let value = processing.buffers.hough[(a * rhos + r) as usize];
            hough_img.put_pixel(a, r, image::Luma([255 - value as u8]));
        }
    }

    let mut corners_img = processing.buffers.source_image.clone();
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();
    for a in 0..angles {
        for r_h in 0..rhos {
            if processing.buffers.hough[(a * rhos + r_h) as usize] > 200 {
                for d_abs in 0..=20 * diagonal as usize {
                    let d = (d_abs as f64 / 10.0) - diagonal;
                    let r = r_h as f64 * diagonal / rhos as f64;
                    let r2 = (r*r + d*d).sqrt();
                    let d2 = (a as f64 * 2.0 * std::f64::consts::PI / angles as f64) - (d / r2).asin();

                    let x = r2 * d2.cos();
                    let y = r2 * d2.sin();

                    if 0.0 <= x && x < width as f64 && 0.0 <= y && y < height as f64 {
                        corners_img.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 255, 255]));
                    }
                }
            }
        }
    }

    for (a, r_h, _) in processing.buffers.lines.iter() {
        for d_abs in 0..=20 * diagonal as usize {
            let d = (d_abs as f64 / 10.0) - diagonal;
            let r = r_h * diagonal / rhos as f64;
            let r2 = (r*r + d*d).sqrt();
            let d2 = (a * 2.0 * std::f64::consts::PI / angles as f64) - (d / r2).asin();

            let x = r2 * d2.cos();
            let y = r2 * d2.sin();

            if 0.0 <= x && x < width as f64 && 0.0 <= y && y < height as f64 {
                corners_img.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 255, 255]));
            }
        }
    }

    for corner in processing.buffers.corners.iter() {
        for dx in -7 ..= 7 {
            for dy in -7 ..= 7 {
                let x = corner.0 as i32 + dx;
                let y = corner.1 as i32 + dy;
                if 0 <= x && x < width as i32 && 0 <= y && y < height as i32 {
                    corners_img.put_pixel(
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

    processing.buffers.source_image.save(format!("outputs/{}.00-original.png", stem)).unwrap();
    sobel_img.save(format!("outputs/{}.01-sobel.png", stem)).unwrap();
    border_img.save(format!("outputs/{}.02-border.png", stem)).unwrap();
    hough_img.save(format!("outputs/{}.03-hough.png", stem)).unwrap();
    corners_img.save(format!("outputs/{}.04-corners.png", stem)).unwrap();
    processing.buffers.perspective_image.save(format!("outputs/{}.05-perspective.png", stem)).unwrap();

    match best {
        Some(entries) => {
            for (i, (entry, score)) in entries.iter().enumerate() {
                let file = std::fs::File::open(&entry.path).unwrap();
                image::io::Reader::new(std::io::BufReader::new(file))
                    .with_guessed_format()
                    .unwrap()
                    .decode()
                    .unwrap()
                    .save(format!("outputs/{}.06-best-{}.png", stem, i))
                    .unwrap();

                println!("match{}: {:?} ({})", i, entry.path, score);
            }
        },
        None => {},
    }
}
