use itertools::Itertools;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        Self {
            layers: layer_sizes
                .iter()
                .copied()
                // Add a 0 value at the back to create a final window with 0
                // weights representing the output layer
                .chain(std::iter::once(0))
                // Window iter used to access both current and next layer sizes
                .tuple_windows()
                .map(|(size, next_size)| Layer::new(size, next_size))
                .collect(),
        }
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
}

pub struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    pub fn new(size: usize, next_size: usize) -> Self {
        Self {
            nodes: Vec::from_iter(std::iter::repeat_with(|| Node::new(next_size)).take(size)),
        }
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

pub struct Node {
    bias: f32,
    weights: Vec<f32>,
}

impl Node {
    pub fn new(weight_count: usize) -> Self {
        Self {
            bias: 0.0,
            weights: vec![0.0; weight_count],
        }
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}
