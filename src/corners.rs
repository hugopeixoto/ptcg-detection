pub fn calculate(hough: &Vec<u32>, width: u32, height: u32) -> Vec<(f64, f64)> {
    let angles = 900;
    let rhos = 900;
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();

    let mut lines = vec![];
    for a in 0..angles {
        for r in 0..rhos {
            if hough[a * rhos + r] > 200 {
                lines.push((a, r));
            }
        }
    }

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

    if final_intersections.len() != 4 {
        return vec![];
    }

    let mut corners = vec![];
    for (x, y) in [(0.0, 0.0), (width as f64, 0.0), (0.0, height as f64), (width as f64, height as f64)] {
        // find closest point to image corner
        corners.push(*final_intersections.iter().min_by_key(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)) as u32).unwrap());
    }

    corners
}
