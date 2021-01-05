extern crate neural_net;
extern crate rand;

use neural_net::matrix::Vector;
use neural_net::neural_net::NeuralNet;

use rand::Rng;

fn main() {
    let mut nn = NeuralNet::new(vec![2, 8, 8, 1]);
    let input = Vector(vec![0.0, 0.0]);
    println!("{:?} -> {:#?}", input, nn);
    let output = nn.feed_forward(input.clone());
    println!("OUTPUT: {:?}", output);
    println!();


    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let mut batch = Vec::new();
        for _ in 0..30 {
            let x = rng.next_f64() * 4.0 - 2.0;
            let y = rng.next_f64() * 4.0 - 2.0;
            let out = if x * x + y * y > 1.0 { 0.0 } else { 1.0 };
            batch.push((Vector(vec![x, y]), Vector(vec![out])));
        }
        nn.mini_batch(batch, 3.0);
    }

    let input = Vector(vec![0.0, 0.0]);
    println!("INPUT: {:?}", input);
    println!("OUTPUT: {:?}", nn.feed_forward(input));

    let input = Vector(vec![0.5, 0.0]);
    println!("INPUT: {:?}", input);
    println!("OUTPUT: {:?}", nn.feed_forward(input));

    let input = Vector(vec![0.0, 0.5]);
    println!("INPUT: {:?}", input);
    println!("OUTPUT: {:?}", nn.feed_forward(input));

    let input = Vector(vec![1.0, 1.0]);
    println!("INPUT: {:?}", input);
    println!("OUTPUT: {:?}", nn.feed_forward(input));

    for x_idx in 0..30 {
        for y_idx in 0..30 {
            let x = (x_idx as f64 / 10.0) - 1.5;
            let y = (y_idx as f64 / 10.0) - 1.5;
            let out = nn.feed_forward(Vector(vec![x, y]));

            println!("{} {} {}", x, y, out.0[0] - 0.5);
        }
    }
}
