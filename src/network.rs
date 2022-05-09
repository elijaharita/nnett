use nalgebra as na;

pub struct Network {
    input_size: na::Vector2<usize>,
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(input_size: na::Vector2<usize>) -> Self {
        Self {
            input_size,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, mut layer: Box<dyn Layer>) {
        layer.set_input_size(
            self.layers
                .last()
                .map(|layer| layer.output_size())
                .unwrap_or(self.input_size),
        );
        self.layers.push(layer);
    }

    pub fn evaluate(&self, input: &[f32]) -> Vec<f32> {
        self.evaluate_and(input, |_, _| ())
    }

    pub fn evaluate_and(&self, input: &[f32], mut f: impl FnMut(&Box<dyn Layer>, &[f32])) -> Vec<f32> {
        let mut input = input;
        let mut output = Vec::new();
        for layer in self.layers.iter() {
            output = layer.evaluate(input);
            f(layer, &output);
            input = &output;
        }

        output
    }

    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn output_size(&self) -> na::Vector2<usize> {
        if let Some(layer) = self.layers.last() {
            layer.output_size()
        } else {
            na::Vector2::zeros()
        }
    }
}

pub trait Layer {
    fn evaluate(&self, input: &[f32]) -> Vec<f32>;
    fn set_input_size(&mut self, input_size: na::Vector2<usize>);
    fn output_size(&self) -> na::Vector2<usize>;
}

pub struct ConvolutionalLayer {
    input_size: na::Vector2<usize>,
    filter_size: na::Vector2<usize>,
    filter_weights: Vec<f32>,
}

impl Layer for ConvolutionalLayer {
    fn evaluate(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_size.product() {
            panic!("wrong input size")
        }

        let output_size = self.output_size();

        let mut output = vec![0.0; output_size.product()];

        for output_x in 0..output_size.x {
            for output_y in 0..output_size.y {
                let output_index = output_x + output_y * output_size.x;

                // NOTE: this loop does not need to range from negative half
                // filter size to positive half filter size, simply iterating
                // through the output size range in the outer loop works out
                // perfectly since the output size is smaller than the input
                // size
                for filter_x in 0..self.filter_size.x {
                    for filter_y in 0..self.filter_size.y {
                        let filter_index = filter_x + filter_y * self.filter_size.x;

                        let input_x = output_x + filter_x;
                        let input_y = output_y + filter_y;
                        let input_index = input_x + input_y * self.input_size.x;
                        output[output_index] +=
                            input[input_index] * self.filter_weights[filter_index];
                    }
                }
            }
        }

        output
    }

    fn set_input_size(&mut self, input_size: na::Vector2<usize>) {
        self.input_size = input_size;
    }

    fn output_size(&self) -> na::Vector2<usize> {
        self.input_size - self.filter_size + na::Vector2::repeat(1)
    }
}

impl ConvolutionalLayer {
    pub fn new(filter_size: na::Vector2<usize>) -> Self {
        Self {
            input_size: na::Vector2::zeros(),
            filter_size,
            filter_weights: vec![0.0; filter_size.product()],
            // filter_weights: std::iter::repeat_with(|| rand::random::<f32>())
            //     .take(filter_size.product())
            //     .collect(),
        }
    }
}

pub struct FullyConnectedLayer {
    input_size: na::Vector2<usize>,
    output_size: na::Vector2<usize>,
    weights: Vec<f32>, // Stored input-major
}

impl Layer for FullyConnectedLayer {
    fn evaluate(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_size.product() {
            panic!("wrong input size");
        }

        let mut output = vec![0.0; self.output_size.product()];
        for input_i in 0..self.input_size.product() {
            for output_i in 0..self.output_size.product() {
                output[output_i] +=
                    input[input_i] * self.weights[self.weight_index(input_i, output_i)];
            }
        }

        output
    }

    fn set_input_size(&mut self, input_size: na::Vector2<usize>) {
        self.input_size = input_size;
        self.weights = vec![0.0; self.input_size.product() * self.output_size.product()];
    }

    fn output_size(&self) -> na::Vector2<usize> {
        self.output_size
    }
}

impl FullyConnectedLayer {
    pub fn new(output_size: na::Vector2<usize>) -> Self {
        Self {
            input_size: na::Vector2::zeros(),
            output_size,
            weights: Vec::new(),
        }
    }

    fn weight_index(&self, input_i: usize, output_i: usize) -> usize {
        input_i + self.input_size.product() * output_i
    }
}

pub struct ReluLayer {
    size: na::Vector2<usize>,
}

impl Layer for ReluLayer {
    fn evaluate(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.size.product() {
            panic!("wrong input size");
        }

        let mut output = vec![0.0; self.size.product()];
        for i in 0..self.size.product() {
            output[i] = input[i].max(0.0);
        }

        output
    }

    fn set_input_size(&mut self, input_size: na::Vector2<usize>) {
        self.size = input_size;
    }

    fn output_size(&self) -> na::Vector2<usize> {
        self.size
    }
}

impl ReluLayer {
    pub fn new() -> Self {
        Self {
            size: na::Vector2::zeros(),
        }
    }
}

pub struct SoftMaxLayer {
    size: na::Vector2<usize>,
}

impl Layer for SoftMaxLayer {
    fn evaluate(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.size.product() {
            panic!("wrong input size");
        }

        let quotient = input
            .iter()
            .map(|&zj| std::f32::consts::E.powf(zj))
            .sum::<f32>();

        input
            .iter()
            .map(|&zi| std::f32::consts::E.powf(zi) / quotient)
            .collect()
    }

    fn set_input_size(&mut self, input_size: na::Vector2<usize>) {
        self.size = input_size;
    }

    fn output_size(&self) -> na::Vector2<usize> {
        self.size
    }
}

impl SoftMaxLayer {
    pub fn new() -> Self {
        Self {
            size: na::Vector2::zeros(),
        }
    }
}
