extern crate neural_net;
extern crate rand;
extern crate byteorder;

use neural_net::matrix::Vector;
use neural_net::neural_net::{NeuralNet, StandardNeuralNet, ActivationFunction, CostFunction};

use rand::Rng;

use byteorder::{BigEndian, ReadBytesExt};

use std::fs::File;

fn main() {
    let mut rng = rand::thread_rng();

    println!("Loading training data.");
    let mut data = load_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", Some(50000));

    println!("Loading testing data.");
    let testing_data = load_data("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", Some(10000));

    println!("Loading validation data.");
    let validation_data = load_data("data/validation-images-idx3-ubyte", "data/validation-labels-idx1-ubyte", None);

    println!("Initializing Neural Network.");
    let mut nn = StandardNeuralNet::new(vec![784, 100, 10]);

    println!("Training Neural Network.");
    for epoch in 0..30 {
        rng.shuffle(&mut data);

        for batch_num in 0..5000 {
            let batch = data[batch_num * 10..(batch_num + 1) * 10].iter().map(|i| i.clone());
            nn.mini_batch(batch, 3.0);
        }

        let num_correct = check_progress(&nn, &testing_data);
        println!("Epoch {} complete. ({} / {})", epoch, num_correct, testing_data.len());

        let test = nn.feed_forward(testing_data[0].0.clone());
        println!("Test case 0 -> {:?}", test);
        println!("{}", output_max(&test));
//        println!("Test case 1 -> {:?}", nn.feed_forward(testing_data[1].0.clone()));
    }

    println!("Validating neural net.");
    for test in validation_data.into_iter() {
        let output = nn.feed_forward(test.0.clone());
        println!("Thought {}. Actually {}.", output_max(&output), output_max(&test.1));
        println!("({:?} -> {:?})", output, test.1);
    }
}

fn check_progress<A, C>(nn: &NeuralNet<A, C>, testing_data: &Vec<(Vector<f64>, Vector<f64>)>) -> u32
    where A: ActivationFunction,
          C: CostFunction,
{
    let mut correct = 0;
    for test in testing_data {
        let output = nn.feed_forward(test.0.clone());
        if output_max(&output) == output_max(&test.1) {
            correct += 1;
        }
    }
    correct
}

fn output_max(vec: &Vector<f64>) -> usize {
    let mut max_idx = 0;
    let mut max_value = -1.0;
    for (idx, value) in vec.0.iter().enumerate() {
        if *value > max_value {
            max_value = *value;
            max_idx = idx;
        }
    }
    max_idx
}

fn load_data(image_path: &str, label_path: &str, limit: Option<usize>) -> Vec<(Vector<f64>, Vector<f64>)> {
    let images = load_images(image_path, limit);
    let labels = load_labels(label_path, limit);

    let images_iter = images.into_iter().map(|i| Vector(i));
    let labels_iter = labels.into_iter().map(|i| {
        let mut out = Vector::new_zero(10);
        out.0[i] = 1.0;
        out
    });

    images_iter.zip(labels_iter).collect()
}

fn load_images(path: &str, limit: Option<usize>) -> Vec<Vec<f64>> {
    let mut image_file = File::open(path).unwrap();

    let magic = image_file.read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 0x00000803);

    let mut images = Vec::new();
    let num_images = image_file.read_u32::<BigEndian>().unwrap();

    let num_images_to_read = limit.unwrap_or(num_images as usize);

    let height = image_file.read_u32::<BigEndian>().unwrap();
    let width = image_file.read_u32::<BigEndian>().unwrap();

    for _ in 0..num_images_to_read {
        let mut image = Vec::new();
        for _ in 0..height {
            for _ in 0..width {
                image.push(image_file.read_u8().unwrap() as f64 / 255.0);
            }
        }
        images.push(image);
    }

    images
}

fn load_labels(path: &str, limit: Option<usize>) -> Vec<usize> {
    let mut label_file = File::open(path).unwrap();

    let magic = label_file.read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 0x00000801);

    let mut labels = Vec::new();
    let num_labels = label_file.read_u32::<BigEndian>().unwrap();

    let num_labels_to_read = limit.unwrap_or(num_labels as usize);

    for _ in 0..num_labels_to_read {
        labels.push(label_file.read_u8().unwrap() as usize);
    }

    labels
}
