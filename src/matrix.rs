use num::Zero;

use std::iter;
use std::ops;

#[derive(Debug,Clone)]
pub struct Matrix<T> {
    elements: Vec<T>,
    rows: usize,
    columns: usize,
}

impl<T> Matrix<T> {
    pub fn new(elements: Vec<T>, rows: usize, columns: usize) -> Matrix<T> {
        if elements.len() != rows * columns {
            panic!("Element size != rows * columns");
        }

        Matrix {
            elements: elements,
            rows: rows,
            columns: columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn elements(&self) -> usize {
        self.elements.len()
    }

    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        // Don't have to check row, since the
        // else branch will return None if row
        // is too large.
        if column >= self.columns {
            None
        }
        else {
            self.elements.get(row * self.columns + column)
        }
    }

    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        // Don't have to check row, since the
        // else branch will return None if row
        // is too large.
        if column >= self.columns {
            None
        }
        else {
            self.elements.get_mut(row * self.columns + column)
        }
    }
}

impl<T> Matrix<T>
    where T: Clone
{
    pub fn transpose(self) -> Matrix<T> {
        let mut transpose = self.elements.clone();
        for idx in 0..self.elements.len() - 1 {
            transpose[(self.rows * idx) % (self.elements.len() - 1)] = self.elements[idx].clone();
        }

        Matrix {
            elements: transpose,
            rows: self.columns,
            columns: self.rows,
        }
    }
}

impl<T> Matrix<T>
    where T: Default + Clone + Zero
{
    pub fn new_zero(rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            elements: iter::repeat(T::zero()).take(rows * columns).collect(),
            rows: rows,
            columns: columns,
        }
    }
}

impl<T> ops::Mul for Matrix<T>
    where T: Copy + ops::Mul<Output = T> + ops::Add + Zero
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.rows {
            panic!("Invalid matrix*matrix dimensions ({},{}) * ({},{})", self.rows, self.columns, rhs.rows, rhs.columns);
        }

        let mut new = Vec::new();

        // naïve matrix multiplication
        for row in 0..self.rows {
            for column in 0..rhs.columns {
                let mut sum = T::zero();
                for idx in 0..self.columns {
                    sum = sum + *self.get(row, idx).unwrap() * *rhs.get(idx, column).unwrap();
                }
                new.push(sum);
            }
        }

        Matrix {
            elements: new,
            rows: self.rows,
            columns: rhs.columns,
        }
    }
}

impl<T> ops::Mul<Vector<T>> for Matrix<T>
    where T: Copy + ops::Mul<Output = T> + ops::Add + Zero
{
    type Output = Vector<T>;
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        if self.columns != rhs.0.len() {
            panic!("Invalid matrix*vector dimensions ({},{}) * ({})", self.rows, self.columns, rhs.0.len());
        }

        let mut new = Vec::new();

        for row in 0..self.rows {
            let mut sum = T::zero();
            for idx in 0..self.columns {
                sum = sum + *self.get(row, idx).unwrap() * *rhs.0.get(idx).unwrap();
            }
            new.push(sum);
        }

        Vector(new)
    }
}

impl<T> ops::Mul<T> for Matrix<T>
    where T: Copy + ops::Mul<Output = T> + Zero
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut new = Vec::new();

        // naïve matrix multiplication
        for row in 0..self.rows {
            for column in 0..self.columns {
                new.push(*self.get(row, column).unwrap() * rhs)
            }
        }

        Matrix {
            elements: new,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<T> ops::Add for Matrix<T>
    where T: Copy + ops::Add<Output = T> + Zero
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.columns || self.rows != rhs.rows {
            panic!("Invalid matrix+matrix dimensions ({},{}) + ({},{})", self.rows, self.columns, rhs.rows, rhs.columns);
        }

        let mut new = Vec::new();

        // naïve matrix multiplication
        for row in 0..self.rows {
            for column in 0..self.columns {
                new.push(*self.get(row, column).unwrap() + *rhs.get(row, column).unwrap());
            }
        }

        Matrix {
            elements: new,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<T> ops::AddAssign for Matrix<T>
    where T: Copy + ops::AddAssign + Zero
{
    fn add_assign(&mut self, rhs: Self) {
        if self.columns != rhs.columns || self.rows != rhs.rows {
            panic!("Invalid matrix+matrix dimensions ({},{}) + ({},{})", self.rows, self.columns, rhs.rows, rhs.columns);
        }

        for element in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *element.0 += *element.1;
        }
    }
}

impl<T> ops::Sub for Matrix<T>
    where T: Copy + ops::Sub<Output = T> + Zero
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.columns || self.rows != rhs.rows {
            panic!("Invalid matrix+matrix dimensions ({},{}) + ({},{})", self.rows, self.columns, rhs.rows, rhs.columns);
        }

        let mut new = Vec::new();

        // naïve matrix multiplication
        for row in 0..self.rows {
            for column in 0..self.columns {
                new.push(*self.get(row, column).unwrap() - *rhs.get(row, column).unwrap());
            }
        }

        Matrix {
            elements: new,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<T> ops::SubAssign for Matrix<T>
    where T: Copy + ops::SubAssign
{
    fn sub_assign(&mut self, rhs: Self) {
        if self.columns != rhs.columns || self.rows != rhs.rows {
            panic!("Invalid matrix+matrix dimensions ({},{}) - ({},{})", self.rows, self.columns, rhs.rows, rhs.columns);
        }

        for element in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *element.0 -= *element.1;
        }
    }
}

#[derive(Debug,Clone)]
pub struct Vector<T>(pub Vec<T>);

impl<T> Vector<T>
    where T: Copy + ops::Mul<Output = T> + Zero
{
    pub fn hadamard(&self, rhs: Vector<T>) -> Vector<T> {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector+vector dimensions ({}) + ({})", self.0.len(), rhs.0.len());
        }

        let new = self.0.iter().zip(rhs.0).map(|t| *t.0 * t.1).collect();

        Vector(new)
    }

    pub fn outer(self, rhs: Vector<T>) -> Matrix<T> {
        let mut m = Vec::new();
        let rows = self.0.len();
        let columns = rhs.0.len();

        for row in self.0.into_iter() {
            for col in rhs.0.iter() {
                m.push(row * *col);
            }
        }

        Matrix {
            elements: m,
            rows: rows,
            columns: columns,
        }
    }

    pub fn map<F, U>(self, func: F) -> Vector<U>
        where F: FnMut(T) -> U
    {
        Vector(self.0.into_iter().map(func).collect())
    }
}

impl<T> Vector<T>
    where T: Default + Clone + Zero
{
    pub fn new_zero(len: usize) -> Vector<T> {
        Vector(iter::repeat(T::zero()).take(len).collect())
    }
}

impl<T> ops::Mul<Matrix<T>> for Vector<T>
    where T: Copy + ops::Mul<Output = T> + ops::Add + Zero
{
    type Output = Vector<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        if self.0.len() != rhs.rows {
            panic!("Invalid vector*matrix dimensions ({}) * ({},{})", self.0.len(), rhs.rows, rhs.columns);
        }

        let mut new = Vec::new();

        for column in 0..self.0.len() {
            let mut sum = T::zero();
            for idx in 0..rhs.rows {
                sum = sum + *self.0.get(column).unwrap() * *rhs.get(idx, column).unwrap();
            }
            new.push(sum);
        }

        Vector(new)
    }
}

impl<T> ops::Mul<Vector<T>> for Vector<T>
    where T: Copy + ops::Mul<Output = T> + ops::Add + Zero
{
    type Output = T;
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector*vector dimensions ({}) * ({})", self.0.len(), rhs.0.len());
        }

        let mut sum = T::zero();
        for idx in 0..rhs.0.len() {
            sum = sum + *self.0.get(idx).unwrap() * *rhs.0.get(idx).unwrap();
        }

        sum
    }
}

impl<T> ops::Mul<T> for Vector<T>
    where T: Copy + ops::Mul<Output = T>
{
    type Output = Vector<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut new = Vec::new();
        for idx in 0..self.0.len() {
            new.push(self.0[idx] * rhs);
        }

        Vector(new)
    }
}

impl<T> ops::Add for Vector<T>
    where T: Copy + ops::Add + Zero
{
    type Output = Vector<T>;
    fn add(self, rhs: Vector<T>) -> Self::Output {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector+vector dimensions ({}) + ({})", self.0.len(), rhs.0.len());
        }

        let new = self.0.into_iter().zip(rhs.0).map(|t| t.0 + t.1).collect();

        Vector(new)
    }
}

impl<T> ops::AddAssign for Vector<T>
    where T: Copy + ops::AddAssign + Zero
{
    fn add_assign(&mut self, rhs: Vector<T>) {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector+vector dimensions ({}) + ({})", self.0.len(), rhs.0.len());
        }

        for element in self.0.iter_mut().zip(rhs.0.iter()) {
            *element.0 += *element.1;
        }
    }
}

impl<T> ops::Sub for Vector<T>
    where T: Copy + ops::Sub<Output = T> + Zero
{
    type Output = Vector<T>;
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector-vector dimensions ({}) + ({})", self.0.len(), rhs.0.len());
        }
        let new = self.0.into_iter().zip(rhs.0).map(|t| t.0 - t.1).collect();

        Vector(new)
    }
}

impl<T> ops::SubAssign for Vector<T>
    where T: Copy + ops::SubAssign
{
    fn sub_assign(&mut self, rhs: Vector<T>) {
        if self.0.len() != rhs.0.len() {
            panic!("Invalid vector-vector dimensions ({}) + ({})", self.0.len(), rhs.0.len());
        }

        for element in self.0.iter_mut().zip(rhs.0.iter()) {
            *element.0 -= *element.1;
        }
    }
}
