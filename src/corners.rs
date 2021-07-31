
fn wrapped_delta(p1: f64, p2: f64, width: f64) -> f64 {
    if p1 > p2 {
        -wrapped_delta(p2, p1, width)
    } else if p2 - p1 < width / 2.0 {
        p2 - p1
    } else {
        p2 - (p1 + width)
    }
}

fn wrapped_weighted_average(p1: f64, c1: f64, p2: f64, c2: f64, width: f64) -> f64 {
    let d = wrapped_delta(p1, p2, width);
    (p1 * c1 + (p1 + d) * c2) / (c1 + c2)
}

pub fn calculate(hough: &Vec<u32>, width: u32, height: u32, lines: &mut Vec<(f64, f64, usize)>, corners: &mut Vec<(f64, f64)>) {
    let angles = 900;
    let rhos = 900;
    let diagonal = ((width * width + height * height) as f64).sqrt().ceil();

    let mut candidate_lines = vec![];
    for a in 0..angles {
        for r in 0..rhos {
            if hough[a * rhos + r] > 200 {
                candidate_lines.push((a as f64, r as f64));
            }
        }
    }

    lines.truncate(0);
    for point in candidate_lines.iter() {
        let dup = lines.iter().position(|p| {
            let delta_angle = wrapped_delta(p.0, point.0, angles as f64).abs();
            let delta_rho = (p.1 - point.1).abs();

            delta_angle < 20.0 && delta_rho < 50.0
        });

        match dup {
            None => {
                lines.push((point.0, point.1, 1));
            },
            Some(index) => {
                let (p, r, c) = lines[index];

                lines[index] = (
                    wrapped_weighted_average(p, c as f64, point.0, 1.0, angles as f64),
                    (r * c as f64 + point.1 as f64) / (c as f64 + 1.0),
                    c + 1,
                );
            }
        }
    }

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
