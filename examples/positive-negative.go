package examples

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Guilherme-De-Marchi/neural-network-studies/perceptron"
)

func PositiveNegative() {
	rand.Seed(time.Now().UnixNano())

	p := perceptron.NewPerceptron(.0001)

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
