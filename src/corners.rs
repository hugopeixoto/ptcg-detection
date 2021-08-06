pub fn calculate(lines: &Vec<(f64, f64, usize)>, width: u32, height: u32, corners: &mut Vec<(f64, f64)>) {
    let angles = 900;
    let rhos = 900;
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();

    let mut intersections = vec![];
    for i1 in 0..lines.len() {
        let p1 = lines[i1];
        for i2 in i1 + 1 .. lines.len() {
            let p2 = lines[i2];
            // find intersection
            let a1 = p1.0 * 2.0 * std::f64::consts::PI / angles as f64;
            let a2 = p2.0 * 2.0 * std::f64::consts::PI / angles as f64;
            let r1 = p1.1;
            let r2 = p2.1;
            let ct1 = a1.cos();
            let ct2 = a2.cos();
            let st1 = a1.sin();
            let st2 = a2.sin();
            let det = ct1 * st2 - st1 * ct2;
            if det != 0.0 {
                let x3 = (st2 * (r1 * diagonal / rhos as f64) - st1 * (r2 * diagonal / rhos as f64)) / det;
                let y3 = (-ct2 * (r1 * diagonal / rhos as f64) + ct1 * (r2 * diagonal / rhos as f64)) / det;

                if 0.0 <= x3 && x3 < width as f64 && 0.0 <= y3 && y3 < height as f64 {
                    intersections.push((x3, y3));
                }
            }
        }
    }

    corners.truncate(0);
    if intersections.len() != 4 {
        return;
    }

    // Sort corners according to proximity to image corners.
    // This assumes the card is not rotated to make processing faster,
    // avoiding having to try several rotations.
    for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
        corners.push(*intersections.iter().min_by_key(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)) as u32).unwrap());
    }
}
