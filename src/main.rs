use std::io::Write;
use std::io::BufRead;
use image::GenericImage;
use image::GenericImageView;
use std::time::Instant;

struct DatasetEntry {
    hash: img_hash::ImageHash,
    header_hash: img_hash::ImageHash,
    art_hash: img_hash::ImageHash,
    path: std::path::PathBuf,
    colors: Vec<(u8, u8, u8)>,
}

const bucket: u8 = 32;
fn colors(image: &image::DynamicImage) -> Vec<(u8, u8, u8)> {
    let mut colors: std::collections::HashMap<(u8,u8,u8), u32> = std::collections::HashMap::new();
    for x in 0..image.width() {
        for y in 0..image.height() {
            let p = image.get_pixel(x, y);
            let r = p[0] / bucket;
            let g = p[1] / bucket;
            let b = p[2] / bucket;

            if colors.contains_key(&(r,g,b)) {
                let v = colors[&(r,g,b)];
                colors.insert((r,g,b), v + 1);
            } else {
                colors.insert((r,g,b), 1);
            }
        }
    }

    let mut sorted = colors.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|&(_, v)| -(*v as i32));
    sorted.truncate(5);

    return sorted.iter().map(|&(k, _)| *k).collect();
}

fn find_border(sobel: &Vec<u32>, width: u32, height: u32) -> Vec<u32> {
    let mut border = vec![];
    border.resize((width*height) as usize, 0);

    let border_threshold = 60;
    let margin = 0;

    for x in margin..width as usize - margin {
        for y in margin..height as usize - margin {
            let magnitude = sobel[x * height as usize + y];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
        for ry in margin..height as usize - margin {
            let y = height as usize - 1 - ry;
            let magnitude = sobel[x * height as usize + y];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
    }
    for y in margin..height as usize - margin {
        for x in margin..width as usize - margin {
            let magnitude = sobel[(x * height as usize + y) as usize];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
        for rx in margin..width as usize - margin {
            let x = width as usize - 1 - rx;
            let magnitude = sobel[(x * height as usize + y) as usize];
            if magnitude > border_threshold {
                border[x * height as usize + y] = 1;
                break;
            }
        }
    }

    border
}

fn calculate_sobel(sobel: &image::DynamicImage, width: u32, height: u32) -> Vec<u32> {
    let luma8 = sobel.to_luma8();
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

    sobel
}

fn calculate_hough(border: &Vec<u32>, width: u32, height: u32) -> Vec<u32> {
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();
    let angles = 900;
    let rhos = 900;

    let mut hough = vec![];
    hough.resize(angles * rhos, 0);

    let mut trigs = Vec::with_capacity(angles);
    for a in 0..angles {
        let theta = a as f64 * 2.0 * std::f64::consts::PI / angles as f64; // t/a = 2pi/400 => t = 2pi/400 * a
        trigs.push((theta.cos(), theta.sin()));
    }

    for x in 0..width {
        for y in 0..height {
            let magnitude = border[(x * height + y) as usize];
            if magnitude > 0 {
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

    hough
}


fn process(source_image: &image::DynamicImage, dataset: &mut Vec<DatasetEntry>, stem: &str) {
    println!(":::::::::::::::::::::::::::::: started");

    let mut time = Instant::now();

    let source_width = source_image.width();
    let source_height = source_image.height();

    let downscaled = source_image.resize(800, 800, image::imageops::FilterType::Triangle);

    eprintln!("[{:?}] downscaled to {:?}", time.elapsed(), downscaled.dimensions());
    time = Instant::now();

    let width = downscaled.width();
    let height = downscaled.height();
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();

    let sobel = calculate_sobel(&downscaled, width, height);
    eprintln!("[{:?}] calculate_sobel", time.elapsed());
    time = Instant::now();

    let border = find_border(&sobel, width, height);
    eprintln!("[{:?}] find_border", time.elapsed());

    let hough = calculate_hough(&border, width, height);
    eprintln!("[{:?}] calculate_hough", time.elapsed());
    time = Instant::now();

    let angles = 900;
    let rhos = 900;

    let mut points = vec![];
    for a in 0..angles {
        for r in 0..rhos {
            let x = hough[a * rhos + r];
            if x > 100 {
                points.push((a, r));
            }
        }
    }

    eprintln!("[{:?}] found lines ({})", time.elapsed(), points.len());
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

    eprintln!("[{:?}] clustered lines ({})", time.elapsed(), clustered_points.len());
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

    eprintln!("[{:?}] found intersections ({})", time.elapsed(), intersections.len());
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

    eprintln!("[{:?}] clustered intersections ({})", time.elapsed(), clustered_intersections.len());
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

    eprintln!("[{:?}] removed parallel intersections ({:?})", time.elapsed(), final_intersections);
    time = Instant::now();

    if final_intersections.len() != 4 {
        return;
    }

    let mut ordered_intersections = vec![];
    for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
        // find closest point to image corner
        ordered_intersections.push(final_intersections.iter().min_by_key(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)) as u32).unwrap());
    }

    eprintln!("[{:?}] sorted intersections ({:?})", time.elapsed(), ordered_intersections);
    time = Instant::now();

    let a3 = nalgebra::Matrix3::new(
        ordered_intersections[0].0, ordered_intersections[1].0, ordered_intersections[2].0,
        ordered_intersections[0].1, ordered_intersections[1].1, ordered_intersections[2].1,
        1.0, 1.0, 1.0,
    );

    let a1 = nalgebra::Vector3::new(ordered_intersections[3].0, ordered_intersections[3].1, 1.0);
    let ax = match a3.lu().solve(&a1) {
        Some(x) => x,
        None => { return; }
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

    let mut perspective_image = image::DynamicImage::new_rgba8(734, 1024);
    for x in 0..734 {
        for y in 0..1024 {
            let p = c * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
            let px = (p[0] / p[2]) as i32;
            let py = (p[1] / p[2]) as i32;

            if 0 <= px && px < width as i32 && 0 <= py && py < height as i32 {
                perspective_image.put_pixel(x, y, source_image.get_pixel(px as u32 * source_width / width, py as u32 * source_height / height));
            }
        }
    }

    eprintln!("[{:?}] perspective!", time.elapsed());
    time = Instant::now();

    let hasher = img_hash::HasherConfig::new().hash_size(16,16).hash_alg(img_hash::HashAlg::Gradient).to_hasher();

    let hash = hasher.hash_image(&perspective_image);
    //let h2 = hasher.hash_image(&perspective_image.clone().crop(0, 0, 734, 90));

    eprintln!("[{:?}] calculate_phash", time.elapsed());
    time = Instant::now();

    let best = dataset.iter().min_by_key(|entry| hash.dist(&entry.hash)).unwrap();

    println!("[{:?}] find_match {:?}", time.elapsed(), best.path);
    time = Instant::now();

    let mut border_image = image::DynamicImage::new_rgba8(width, height);
    for x in 0..width as usize {
        for y in 0..height as usize {
            if border[x * height as usize + y] > 0 {
                border_image.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 0, 255]));
            }
        }
    }

    let mut lines_image = source_image.clone();
    for &(a, r_h) in points.iter() {
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


    image::open(format!("images/canon/{}.png", stem)).unwrap().save(format!("outputs/{}.original.png", stem)).unwrap();
    border_image.save(format!("outputs/{}.border.png", stem)).unwrap();
    lines_image.save(format!("outputs/{}.hough-lines.png", stem)).unwrap();
    perspective_image.save(format!("outputs/{}.perspective.png", stem)).unwrap();
    image::open(&best.path).unwrap().save(format!("outputs/{}.best.png", stem)).unwrap();
    println!("[{:?}] debug_images", time.elapsed());
    // let colors = colors(&perspective_image);
    // eprintln!("[{:?}] best color: {:?}", time.elapsed(), colors[0]);
    //
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
}

fn main() {
    let dataset_paths = std::fs::read_dir("dataset/").unwrap();

    let mut dataset = vec![];
    if std::path::Path::new("dataset.txt").exists() {
        for line in std::io::BufReader::new(std::fs::File::open("dataset.txt").unwrap()).lines() {
            let x = line.unwrap();
            let parts = x.split(' ').collect::<Vec<_>>();

            let color_bytes = base64::decode(&parts[4]).unwrap();
            dataset.push(DatasetEntry {
                    path: std::path::PathBuf::from(&parts[0]),
                    hash: img_hash::ImageHash::from_base64(&parts[1]).unwrap(),
                    header_hash: img_hash::ImageHash::from_base64(&parts[2]).unwrap(),
                    art_hash: img_hash::ImageHash::from_base64(&parts[3]).unwrap(),
                    colors: vec![
                        (color_bytes[0], color_bytes[1], color_bytes[2]),
                        (color_bytes[3], color_bytes[4], color_bytes[5]),
                        (color_bytes[6], color_bytes[7], color_bytes[8]),
                        (color_bytes[9], color_bytes[10], color_bytes[11]),
                        (color_bytes[12], color_bytes[13], color_bytes[14]),
                    ],
            })
        }
    } else {
        let mut progress = 0;
        let mut errors = 0;
        let hasher = img_hash::HasherConfig::new().hash_size(16,16).hash_alg(img_hash::HashAlg::Gradient).to_hasher();
        let mut file = std::fs::File::create("dataset.txt").unwrap();
        for entry in dataset_paths {
            progress += 1;
            let path = entry.unwrap().path();
            match image::open(&path) {
                Ok(img) => {
                    let hash = hasher.hash_image(&img);
                    let header_hash = hasher.hash_image(&img.clone().crop(0, 0, 734, 90));
                    let art_hash = hasher.hash_image(&img.clone().crop(60, 100, 615, 380));
                    let colors = colors(&img);
                    let color_bytes = vec![
                        colors[0].0, colors[0].1, colors[0].2,
                        colors[1].0, colors[1].1, colors[1].2,
                        colors[2].0, colors[2].1, colors[2].2,
                        colors[3].0, colors[3].1, colors[3].2,
                        colors[4].0, colors[4].1, colors[4].2,
                    ];
                    file.write(format!("{} {} {} {} {}\n",
                                       path.to_str().unwrap(),
                                       hash.to_base64(),
                                       header_hash.to_base64(),
                                       art_hash.to_base64(),
                                       base64::encode(&color_bytes),
                    ).as_bytes()).unwrap();
                    dataset.push(DatasetEntry { hash, header_hash, art_hash, path, colors });
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
    }

    let paths = std::fs::read_dir("images/canon/").unwrap();

    for path in paths {
        let filename = path.unwrap().path();
        let source_image = image::open(&filename).expect("failed to read image");
        let downscaled = source_image.resize(800, 800, image::imageops::FilterType::Triangle);

        let time = Instant::now();

        process(&downscaled, &mut dataset, filename.file_stem().unwrap().to_str().unwrap());

        eprintln!("[{:?}] processed {:?}", time.elapsed(), filename);
    }
}
