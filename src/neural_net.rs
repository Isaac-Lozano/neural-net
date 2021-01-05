use matrix::{Matrix, Vector};

use rand::distributions::normal::StandardNormal;
use rand;

use std::f64;
use std::marker::PhantomData;

#[derive(Debug,Clone)]
pub struct NeuralNetLayer {
    biases: Vector<f64>,
    weights: Matrix<f64>,
}

impl NeuralNetLayer {
    pub fn from_raw(biases: Vector<f64>, weights: Matrix<f64>) -> NeuralNetLayer {
        NeuralNetLayer {
            biases: biases,
            weights: weights,
        }
    }

    fn new(current: usize, prev: usize) -> NeuralNetLayer {
        let mut layer = Vec::new();
        for _ in 0..current {
            let StandardNormal(bias) = rand::random();
            layer.push(bias);
        }
        let biases = Vector(layer);

        let mut weight_vec = Vec::new();
        for _row in 0..current*prev {
            let StandardNormal(weight) = rand::random();
            weight_vec.push(weight);
        }
        let weights = Matrix::new(weight_vec, current, prev);

        NeuralNetLayer {
            biases: biases,
            weights: weights,
        }
    }

    fn feed_forward<A>(&self, input: Vector<f64>) -> Vector<f64>
        where A: ActivationFunction
    {
        let z = self.weights.clone() * input + self.biases.clone();
        z.map(A::activation)
    }

    fn feed_forward_backprop<A>(&self, input: Vector<f64>) -> (Vector<f64>, Vector<f64>)
        where A: ActivationFunction
    {
        let z = self.weights.clone() * input + self.biases.clone();
        let activation = z.clone().map(A::activation);

        (z, activation)
    }
}

#[derive(Debug,Clone)]
pub struct NeuralNet<A, C> {
    layers: Vec<NeuralNetLayer>,
    activation: PhantomData<A>,
    cost: PhantomData<C>,
}

impl<A, C> NeuralNet<A, C>
    where A: ActivationFunction,
          C: CostFunction,
{
    pub fn new(sizes: Vec<usize>) -> NeuralNet<A, C> {
        let layers = sizes.iter().zip(sizes.iter().skip(1)).map(|(prev, current)|
            NeuralNetLayer::new(*current, *prev)
        ).collect();

        NeuralNet {
            layers: layers,
            activation: PhantomData,
            cost: PhantomData,
        }
    }

    pub fn from_raw(layers: Vec<NeuralNetLayer>) -> NeuralNet<A, C> {
        NeuralNet {
            layers: layers,
            activation: PhantomData,
            cost: PhantomData,
        }
    }

    pub fn feed_forward(&self, input: Vector<f64>) -> Vector<f64> {
        self.layers.iter().fold(input, |inp, layer| {
            layer.feed_forward::<A>(inp)
        })
    }

    pub fn mini_batch<I>(&mut self, batch: I, eta: f64)
        where I: IntoIterator<Item = (Vector<f64>, Vector<f64>)>
    {
        let mut nabla_b: Vec<_> = self.layers.iter().map(|l| Vector::new_zero(l.biases.0.len())).collect();
        let mut nabla_w: Vec<_> = self.layers.iter().map(|l| Matrix::new_zero(l.weights.rows(), l.weights.columns())).collect();
        let mut batch_len = 0;

        for (input, expected) in batch {
            let delta_nablas = self.backprop(input, expected);

            for ((nb, nw), dn) in nabla_b.iter_mut().zip(nabla_w.iter_mut()).zip(delta_nablas.into_iter()) {
                *nb += dn.0;
                *nw += dn.1;
            }

            batch_len += 1;
        }

        for (l, (nb, nw)) in self.layers.iter_mut().zip(nabla_b.into_iter().zip(nabla_w.into_iter())) {
            l.biases -= nb * (eta / batch_len as f64);
            l.weights -= nw * (eta / batch_len as f64);
        }
    }

    fn backprop(&self, input: Vector<f64>, expected: Vector<f64>) -> Vec<(Vector<f64>, Matrix<f64>)> {
        let (mut nablas, _) = self.backprop_internal(input, expected, 0);
        nablas.reverse();
        nablas
    }

    fn backprop_internal(&self, prev_activation: Vector<f64>, expected: Vector<f64>, layer: usize) -> (Vec<(Vector<f64>, Matrix<f64>)>, Vector<f64>) {
        let delta;
        let mut nablas;

        let (z, activation) = self.layers[layer].feed_forward_backprop::<A>(prev_activation.clone());

        if layer == self.layers.len() - 1 {
            delta = C::cost_gradient(&activation, &expected).hadamard(z.map(A::activation_prime));
            nablas = Vec::new();
        }
        else
        {
            let (n, d) = self.backprop_internal(activation, expected, layer + 1);
            delta = (self.layers[layer + 1].weights.clone().transpose() * d).hadamard(z.map(A::activation_prime));
            nablas = n;
        }

        let nabla_b = delta.clone();
        let nabla_w = delta.clone().outer(prev_activation);

        nablas.push((nabla_b, nabla_w));

        (nablas, delta)
    }
}

pub type StandardNeuralNet = NeuralNet<Sigmoid, MeanSquareError>;

pub trait CostFunction {
    fn cost_gradient(output: &Vector<f64>, expected: &Vector<f64>) -> Vector<f64>;
}

pub struct MeanSquareError;

impl CostFunction for MeanSquareError {
    fn cost_gradient(output: &Vector<f64>, expected: &Vector<f64>) -> Vector<f64> {
        output.clone() - expected.clone()
    }
}

pub trait ActivationFunction {
    fn activation(input: f64) -> f64;
    fn activation_prime(input: f64) -> f64;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid{
    fn activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn activation_prime(x: f64) -> f64 {
        let s = Self::activation(x);
        s * (1.0 - s)
    }
}

