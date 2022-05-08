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
        let mut input = input;
        let mut output = Vec::new();
        for layer in self.layers.iter() {
            output = layer.evaluate(input);
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

                output[output_index] /= self.filter_size.product() as f32;
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
            // filter_weights: vec![0.0; filter_size.product()],
            filter_weights: std::iter::repeat_with(|| rand::random::<f32>())
                .take(filter_size.product())
                .collect(),
        }
    }
}
