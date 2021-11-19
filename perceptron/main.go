package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
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

func main() {
	rand.Seed(time.Now().UnixNano())

	p := NewPerceptron(.0001)

	// Training sequence
	for i := 0; i < 10000; i++ {
		v := float64(rand.Intn(1000+1000) - 1000)
		var y float64
		// Positive numbers return 1, and negative, 0
		if v >= 0 {
			y = 1
		} else {
			y = 0
		}

		p.Train(v, y)

	}
	for {
		var v float64
		fmt.Scan(&v)
		var r string
		var accuracy float64
		prediction := p.Predict(v)
		if prediction >= .5 {
			r = "positive"
			accuracy = prediction * 100
		} else {
			r = "negative"
			accuracy = (1 - prediction) * 100
		}
		fmt.Println("it's", r, "\nAccuracy:", accuracy, "%\n")
	}
}
