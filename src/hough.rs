pub fn calculate(border: &Vec<u32>, width: u32, height: u32, hough: &mut Vec<u32>) {
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();
    let angles = 900;
    let rhos = 900;

    for i in hough.iter_mut() { *i = 0; }

    let mut trigs = Vec::with_capacity(angles);
    for a in 0..angles {
        // t/a = 2pi/400 => t = 2pi/400 * a
        let theta = a as f64 * 2.0 * std::f64::consts::PI / angles as f64;
        trigs.push((theta.cos(), theta.sin()));
    }

    for y in 0..height {
        for x in 0..width {
            let magnitude = border[(y * width + x) as usize];
            if magnitude > 0 {
                for a in 0..angles {
                    let rho = x as f64 * trigs[a].0 + y as f64 * trigs[a].1;
                    if rho >= 0.0 {
                        let r = (rho * rhos as f64 / diagonal) as usize;
                        hough[a * rhos + r] += magnitude as u32;
                    }
                }
            }
        }
    }
}
