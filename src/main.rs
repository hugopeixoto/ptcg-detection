use image::Luma;
use image::Rgba;
use image::GenericImage;
use image::GenericImageView;
use std::time::Instant;

fn process(filename: &std::path::PathBuf, dataset: &mut Vec<(img_hash::ImageHash, image::DynamicImage, std::path::PathBuf)>) {
    let stem = filename.file_stem().unwrap().to_str().unwrap();
    println!(":::::::::::::::::::::::::::::: started {}", stem);

    let mut time = Instant::now();

    let source_image = image::open(filename).expect("failed to read image");

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

    sobel_image.save(format!("outputs/{}.sobel.png", stem)).unwrap();
    downscaled.grayscale().save(format!("outputs/{}.grayscale.png", stem)).unwrap();

    let mut canny = image::DynamicImage::new_luma8(width, height).to_luma8();
    for x in 0..width {
        for y in 0..height {
            let mag = (detection[(x as usize,y as usize)].magnitude() * 500.0) as u8;
            canny.put_pixel(x, y, Luma([mag]));
        }
    }

    canny.save(format!("outputs/{}.canny.png", stem)).unwrap();

    println!("[{:?}] saved intermediate images", time.elapsed());
    time = Instant::now();

    let mut border = vec![];
    border.resize((width*height) as usize, 0);
    let border_threshold = 100;
    for x in 10..width as usize {
        for y in 10..height as usize-50 {
            let magnitude = sobel[x * height as usize + y];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
        for ry in 50..height as usize {
            let y = height as usize - 1 - ry;
            let magnitude = sobel[x * height as usize + y];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
    }
    for y in 10..height as usize {
        for x in 10..width as usize {
            let magnitude = sobel[(x * height as usize + y) as usize];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
        for rx in 10..width as usize {
            let x = width as usize - 1 - rx;
            let magnitude = sobel[(x * height as usize + y) as usize];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
    }

    println!("[{:?}] ray casting border", time.elapsed());
    time = Instant::now();

    let mut border_image = image::DynamicImage::new_rgba8(width, height);
    for x in 10..width as usize {
        for y in 10..height as usize -50 {
            if border[x * height as usize + y] > 0 {
                border_image.put_pixel(x as u32, y as u32, Rgba([0, 0, 0, 255]));
            }
        }
    }

    border_image.save(format!("outputs/{}.border.png", stem)).unwrap();
    println!("[{:?}] saved border", time.elapsed());
    time = Instant::now();

    let angles = 300;
    let rhos = 300;

    let mut hough: Vec<u32> = Vec::new();
    hough.resize(angles * rhos, 0);

    let mut trigs = Vec::with_capacity(angles);
    for a in 0..angles {
        let theta = a as f64 * 2.0 * std::f64::consts::PI / angles as f64; // t/a = 2pi/400 => t = 2pi/400 * a
        trigs.push((theta.cos(), theta.sin()));
    }

    for x in 0..width {
        for y in 0..height {
            let magnitude = border[(x * height + y) as usize];
            if magnitude >= 0 {
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
            let x = hough[a * rhos + r]; // (hough[a*rhos + r] * 255 / maximum_weight) as u8;
            if x > 100 {
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
    hough_image.save(format!("outputs/{}.hough.png", stem)).unwrap();

    println!("[{:?}] saved hough transform", time.elapsed());
    time = Instant::now();

    let mut clustered_points: Vec<(f64, f64, usize)> = vec![];
    for point in points.iter() {
        let dup = clustered_points.iter().position(|p| ((p.0/p.2 as f64) - point.0 as f64).abs() < 20.0 && ((p.1/p.2 as f64) - point.1 as f64).abs() < 50.0);
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
    println!("{:?}", clustered_points);
    time = Instant::now();

    let mut intersections = vec![];
    for i1 in 0..clustered_points.len() {
        let p1 = clustered_points[i1];
        for i2 in i1 + 1 .. clustered_points.len() {
            let p2 = clustered_points[i2];
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

    println!("[{:?}] found intersections ({})", time.elapsed(), intersections.len());
    time = Instant::now();

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

    println!("[{:?}] clustered intersections ({})", time.elapsed(), clustered_intersections.len());
    time = Instant::now();

    let mut redundant = vec![];
    redundant.resize(clustered_intersections.len(), false);
    for i in 0..clustered_intersections.len() {
        let p1 = clustered_intersections[i];
        for j in i+1..clustered_intersections.len() {
            let p2 = clustered_intersections[j];
            for k in j+1..clustered_intersections.len() {
                let p3 = clustered_intersections[k];

                let area = 0.5 * (p1.0 * (p2.1 - p3.1) + p2.0 * (p3.1 - p1.1) + p3.0 * (p1.1 - p2.1)).abs();
                if area < 150.0 {
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

    println!("redundant: {:?}", redundant);

    println!("[{:?}] removed parallel intersections ({:?})", time.elapsed(), final_intersections);
    time = Instant::now();

    let mut ordered_intersections = vec![];
    for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
        // find closest point to image corner
        ordered_intersections.push(final_intersections.iter().min_by_key(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)) as u32).unwrap());
    }

    println!("[{:?}] sorted intersections ({:?})", time.elapsed(), ordered_intersections);
    time = Instant::now();

    let mut lines_image = downscaled.clone();
    for (a_s, r_s, c) in clustered_points.iter() {
        if *c == 0 { continue; }
        let a = a_s / *c as f64;
        let r_h = r_s / *c as f64;
        for d_abs in 0..=(20*diagonal as usize) {
            let d = (d_abs as f64 / 10.0) - diagonal;
            let r = r_h * diagonal / rhos as f64;
            let r2 = (r*r + d*d).sqrt();
            let d2 = (a as f64 * 2.0 * std::f64::consts::PI / angles as f64) - (d / r2).asin();

            let x = r2 * d2.cos();
            let y = r2 * d2.sin();

            if 0.0 <= x && x < width as f64 && 0.0 <= y && y < height as f64 {
                lines_image.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 255, 255]));
            }
        }
    }

    for i in ordered_intersections.iter() {
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
    println!("[{:?}] rendered lines", time.elapsed());

    let a3 = nalgebra::Matrix3::new(
        ordered_intersections[0].0, ordered_intersections[1].0, ordered_intersections[2].0,
        ordered_intersections[0].1, ordered_intersections[1].1, ordered_intersections[2].1,
        1.0, 1.0, 1.0,
    );

    let a1 = nalgebra::Vector3::new(ordered_intersections[3].0, ordered_intersections[3].1, 1.0);
    let ax = a3.lu().solve(&a1).unwrap();
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

    //println!("a = {:?}", a);
    //println!("b = {:?}", b);
    //println!("c = {:?}", c);

    let mut perspective_image = image::DynamicImage::new_rgba8(734, 1024);
    for x in 0..734 {
        for y in 0..1024 {
            let p = c * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
            let px = (p[0] / p[2]) as i32;
            let py = (p[1] / p[2]) as i32;

            //println!("{:?}: {}, {}", p, px, py);

            if 0 <= px && px < height as i32 && 0 <= py && py < width as i32 {
                perspective_image.put_pixel(x, y, downscaled.get_pixel(px as u32, py as u32));
            }
        }
    }
    perspective_image.save(format!("outputs/{}.perspective.png", stem)).unwrap();
    println!("[{:?}] perspective!", time.elapsed());

    let hash = img_hash::HasherConfig::new().to_hasher().hash_image(&perspective_image);

    println!("hash: {}", hash.to_base64());

    dataset.sort_by_key(|entry| hash.dist(&entry.0));

    println!("best: {:?} ({})", dataset[0].2, hash.dist(&dataset[0].0));
    dataset[0].1.save(format!("outputs/{}.best.png", stem)).unwrap();

    println!("second: {:?} ({})", dataset[1].2, hash.dist(&dataset[1].0));
    dataset[1].1.save(format!("outputs/{}.second.png", stem)).unwrap();

    println!("third: {:?} ({})", dataset[2].2, hash.dist(&dataset[2].0));
    dataset[2].1.save(format!("outputs/{}.third.png", stem)).unwrap();
}

fn main() {
    let dataset_paths = std::fs::read_dir("dataset/").unwrap();

    let mut progress = 0;
    let mut errors = 0;
    let mut dataset = vec![];
    for path in dataset_paths {
        progress += 1;
        let filename = path.unwrap().path();
        match image::open(&filename) {
            Ok(img) => {
                let hash = img_hash::HasherConfig::new().to_hasher().hash_image(&img);
                dataset.push((hash, img, filename));
            },
            Err(_) => {
                errors += 1;
            }
        }

        if progress % 1000 == 0 {
            println!("loading dataset: {}", progress);
        }
    }
    println!("errors: {}", errors);


    let paths = std::fs::read_dir("images/flash/").unwrap();

    for path in paths {
        process(&path.unwrap().path(), &mut dataset);
    }
}
