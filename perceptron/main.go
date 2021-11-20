package perceptron

import (
	"math"
	"math/rand"
)

type Perceptron struct {
	Weight, Bias, LearningRate float64
}

func NewPerceptron(learning_rate float64) *Perceptron {
	return &Perceptron{rand.Float64(), rand.Float64(), learning_rate}
}

func (p *Perceptron) Predict(x float64) float64 {
	/*
		return a linear function applied on a sigmoid function
		y = 1/[1+e^-(wx+b)]
	*/

	return 1 / (1 + math.Pow(math.E, -(p.Weight*x+p.Bias)))
}

func (p *Perceptron) Train(x, y float64) {
	/*y represents the expected value for the prediction of x*/

	prediction := p.Predict(x)

	//cost := math.Pow((y - prediction), 2)
	p.Weight = p.Weight - p.LearningRate*(-x*(y-prediction)) // w = w-learning_rate{-x[y-y_predicted]}
	p.Bias = p.Bias - p.LearningRate*-(y-prediction)         // w = w-learning_rate[-(y-y_predicted)]
}
