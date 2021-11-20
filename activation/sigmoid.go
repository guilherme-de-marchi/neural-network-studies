package activation

import "math"

func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}
