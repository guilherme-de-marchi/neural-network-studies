package examples

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Guilherme-De-Marchi/neural-network-studies/activation"
	"github.com/Guilherme-De-Marchi/neural-network-studies/perceptron"
)

func MultiPosNeg() {
	rand.Seed(time.Now().UnixNano())

	p := perceptron.NewPerceptron(2, .0001)

	// Training sequence
	for i := 0; i < 100000; i++ {
		a := float64(rand.Intn(1000+1000) - 1000)
		b := float64(rand.Intn(1000+1000) - 1000)

		var y float64
		if a+b >= 0 {
			y = 1
		} else {
			y = 0
		}
		p.Train([]float64{a, b}, y, activation.Sigmoid)
		//fmt.Println(a, b, a+b)
	}
	for {
		var r string
		var accuracy float64

		var a, b float64
		fmt.Scan(&a, &b)

		prediction := p.Predict([]float64{a, b}, activation.Sigmoid)
		if prediction >= .5 {
			r = "positive"
			accuracy = prediction * 100
		} else {
			r = "negative"
			accuracy = (1 - prediction) * 100
		}

		fmt.Println(a, "+", b, "is", r, "| Accuracy:", accuracy, "%\n")
	}
}
