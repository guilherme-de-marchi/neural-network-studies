package perceptron

import (
	"log"
	"math/rand"
)

type Perceptron struct {
	Weights            []float64
	Bias, LearningRate float64
}

func NewPerceptron(qtdWeights int, learning_rate float64) *Perceptron {
	weights := make([]float64, qtdWeights)
	for i := range weights {
		weights[i] = rand.Float64()
	}

	return &Perceptron{weights, rand.Float64(), learning_rate}
}

func (p *Perceptron) Predict(inputs []float64, activationFunc func(float64) float64) float64 {
	/*
		z = sum(wi*xi)+b
		y = 1/[1+e^-z]
	*/

	if len(inputs) != len(p.Weights) {
		log.Fatal("inputs lenght != weights lenght")
	}

	var linearSum float64
	for i, wi := range p.Weights {
		linearSum += wi * inputs[i]
	}
	linearSum += p.Bias

	return activationFunc(linearSum)
}

func (p *Perceptron) Train(inputs []float64, y float64, activationFunc func(float64) float64) {
	/*y represents the expected value for the prediction of inputs*/

	prediction := p.Predict(inputs, activationFunc)
	//fmt.Println(inputs, p.Weights, p.Bias, prediction, math.Pow((prediction-y), 2))

	//cost := math.Pow((prediction - y), 2)
	for i := range p.Weights {
		p.Weights[i] -= p.LearningRate * (-inputs[i] * (y - prediction))
	}
	//p.Weight = p.Weight - p.LearningRate*(-x*(y-prediction)) // w = w-learning_rate{-x[y-y_predicted]}
	p.Bias -= p.LearningRate * -(y - prediction) // w = w-learning_rate[-(y-y_predicted)]
}
