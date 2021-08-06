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

pub fn calculate(hough: &Vec<u32>, lines: &mut Vec<(f64, f64, usize)>) {
    let angles = 900;
    let rhos = 900;

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
}
