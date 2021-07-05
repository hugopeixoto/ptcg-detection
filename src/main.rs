use image::Luma;
use image::GenericImage;
use image::GenericImageView;
use std::time::Instant;

fn main() {
    println!("started");
    let mut time = Instant::now();

    let source_image = image::open("scaled_img_20210705_114314.jpg")
        .expect("failed to read image");

    println!("[{:?}] loaded", time.elapsed());
    time = Instant::now();

    let downscaled = source_image.resize(800, 800, image::imageops::FilterType::Triangle);

    println!("[{:?}] downscaled to {:?}", time.elapsed(), downscaled.dimensions());
    time = Instant::now();

    let width = downscaled.width();
    let height = downscaled.height();
    let diagonal = ((width*width + height*height) as f64).sqrt().ceil();

    let detection = edge_detection::canny(
        downscaled.to_luma8(),
        1.2,  // sigma
        0.1,  // strong threshold
        0.01, // weak threshold
    );

    println!("[{:?}] canny detected", time.elapsed());
    time = Instant::now();

    // sobel

    let luma8 = downscaled.to_luma8();
    let mut sobel_image = image::DynamicImage::new_luma8(width, height).to_luma8();
    let mut sobel = vec![];
    sobel.resize((width * height) as usize, 0);
    for x in 1..width-1 {
        for y in 1..height-1 {
            let val0 = luma8.get_pixel(x - 1, y - 1).0[0] as i32;
            let val1 = luma8.get_pixel(x + 0, y - 1).0[0] as i32;
            let val2 = luma8.get_pixel(x + 1, y - 1).0[0] as i32;
            let val3 = luma8.get_pixel(x - 1, y + 0).0[0] as i32;
            //let val4 = luma8.get_pixel(x +0, y + 0).0[0] as i32;
            let val5 = luma8.get_pixel(x + 1, y + 0).0[0] as i32;
            let val6 = luma8.get_pixel(x - 1, y + 1).0[0] as i32;
            let val7 = luma8.get_pixel(x + 0, y + 1).0[0] as i32;
            let val8 = luma8.get_pixel(x + 1, y + 1).0[0] as i32;

            let gx = -val0 + -2*val3 + -val6 + val2 + 2*val5 + val8;
            let gy = -val0 + -2*val1 + -val2 + val6 + 2*val7 + val8;
            let mag = ((gx*gx + gy*gy) as f64).sqrt().min(255.0);

            sobel[(x*height+y) as usize] = mag as u32;
        }
    }

    println!("[{:?}] sobel operated ({}, {})", time.elapsed(), sobel.iter().max().unwrap(), sobel.iter().filter(|&x| *x == 255).count());
    time = Instant::now();

    for x in 0..width {
        for y in 0..height {
            sobel_image.put_pixel(x, y, Luma([sobel[(x*height+y) as usize] as u8]));
        }
    }

    sobel_image.save("sobel.png").unwrap();
    downscaled.grayscale().save("grayscale.png").unwrap();
    detection.as_image().save("result.png").unwrap();
    detection.as_image().grayscale().save("result-gs.png").unwrap();

    println!("[{:?}] saved intermediate images", time.elapsed());
    time = Instant::now();

    let angles = 400;
    let rhos = 400;

    let mut hough: Vec<u32> = Vec::new();
    hough.resize(angles * rhos, 0);

    let mut trigs = Vec::with_capacity(angles);
    for a in 0..angles {
        let theta = a as f64 * 2.0 * std::f64::consts::PI / angles as f64; // t/a = 2pi/400 => t = 2pi/400 * a
        trigs.push((theta.cos(), theta.sin()));
    }

    for x in 0..width {
        for y in 0..height {
            let magnitude = sobel[(x * height + y) as usize];
            if magnitude == 255 {
                for a in 0..angles {
                    let rho = x as f64 * trigs[a].0 + y as f64 * trigs[a].1;
                    if rho >= 0.0 {
                        let r = (rho * rhos as f64 / diagonal) as usize; // r/rho = 400 / ? => r = 400 / ? * rho
                        hough[a * rhos + r] += magnitude as u32; //as u32 * 100;
                    }
                }
            }
        }
    }

    let maximum_weight = hough.iter().max().unwrap();

    println!("[{:?}] hough transformed ({})", time.elapsed(), maximum_weight);
    time = Instant::now();

    let mut points = vec![];

    for a in 0..angles {
        for r in 0..rhos {
            let x = (hough[a*rhos + r] * 255 / maximum_weight) as u8;
            if x > 180 {
                points.push((a, r));
            }
        }
    }

    println!("[{:?}] found lines ({})", time.elapsed(), points.len());
    time = Instant::now();

    let mut hough_image = image::DynamicImage::new_luma8(angles as u32, rhos as u32);
    for a in 0..angles {
        for r in 0..rhos {
            let x = (hough[a*rhos + r] * 255 / maximum_weight) as u8;
            hough_image.put_pixel(a as u32, r as u32,  image::Rgba([x, x, x, 255]));
        }
    }
    hough_image.save("hough.png").unwrap();

    println!("[{:?}] saved hough transform", time.elapsed());
    time = Instant::now();

    let mut clustered_points: Vec<(f64, f64, usize)> = vec![];
    for point in points.iter() {
        let dup = clustered_points.iter().position(|p| ((p.0/p.2 as f64) - point.0 as f64).abs() < 5.0 && ((p.1/p.2 as f64) - point.1 as f64).abs() < 5.0);
        match dup {
            None => {
                clustered_points.push((point.0 as f64, point.1 as f64, 1));
            },
            Some(index) => {
                let (p, r, c) = clustered_points[index];
                clustered_points[index] = (p + point.0 as f64, r + point.1 as f64, c + 1);
            }
        }
    }

    println!("[{:?}] clustered lines ({})", time.elapsed(), clustered_points.len());
    time = Instant::now();


    let mut lines_image = downscaled.clone();
    for (a_s, r_s, c) in clustered_points.iter() {
        let a = a_s / *c as f64;
        let r_h = r_s / *c as f64;
        for d_abs in 0..=(20*diagonal as usize) {
            let d = (d_abs as f64 / 10.0) - diagonal;
            let r = r_h * diagonal / rhos as f64;
            let r2 = (r*r + d*d).sqrt();
            let d2 = (a as f64 * 2.0 * std::f64::consts::PI / angles as f64) - (d / r2).asin();

            let x = r2 * d2.cos();
            let y = r2 * d2.sin();

            //println!("r2 d2: {} {}, x y: {} {}", r2, d2, x, y);

            if 0.0 <= x && x < width as f64 && 0.0 <= y && y < height as f64 {
                lines_image.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 255, 255]));
            }
        }
    }

    lines_image.save("hough-lines.png").unwrap();

    println!("[{:?}] rendered lines", time.elapsed());
}
