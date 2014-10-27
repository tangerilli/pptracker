package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/lazywei/go-opencv/opencv"
	"os"
)

type Params struct {
	LowH  float64
	LowS  float64
	LowV  float64
	HighH float64
	HighS float64
	HighV float64
}

func (p *Params) Print() {
	fmt.Printf("Low=(%f, %f, %f) High=(%f, %f, %f)\n", p.LowH, p.LowS, p.LowV, p.HighH, p.HighS, p.HighV)
}

func loadParams(path string) (*Params, error) {
	params := Params{0, 0, 0, 179, 255, 255}

	cfgFile, err := os.Open(path)
	if err != nil {
		return &params, nil
	}
	defer cfgFile.Close()

	decoder := json.NewDecoder(cfgFile)
	err = decoder.Decode(&params)
	if err != nil {
		return nil, err
	}
	return &params, nil
}

func writeParams(path string, params *Params) error {
	cfgFile, err := os.Create(path)
	if err != nil {
		return err
	}
	defer cfgFile.Close()

	encoded, err := json.MarshalIndent(params, "", "    ")
	if err != nil {
		return err
	}
	cfgFile.Write(encoded)
	return nil
}

func FindFaces(img *opencv.IplImage, win *opencv.Window) error {
	cascade := opencv.LoadHaarClassifierCascade("haarcascade_frontalface_alt.xml")
	faces := cascade.DetectObjects(img)

	for _, value := range faces {
		opencv.Rectangle(img,
			opencv.Point{value.X() + value.Width(), value.Y()},
			opencv.Point{value.X(), value.Y() + value.Height()},
			opencv.ScalarAll(255.0), 1, 1, 0)
	}
	win.ShowImage(img)
	return nil
}

type BallResult struct {
	Found          bool
	X              int
	Y              int
	ProcessedImage *opencv.IplImage
}

func FindBall(img *opencv.IplImage, win *opencv.Window, params *Params) (BallResult, error) {
	w := img.Width()
	h := img.Height()
	imgHSV := opencv.CreateImage(w, h, opencv.IPL_DEPTH_8U, 3)
	opencv.CvtColor(img, imgHSV, opencv.CV_BGR2HSV)

	// threshold the image
	imgThresholded := opencv.CreateImage(w, h, opencv.IPL_DEPTH_8U, 1)
	opencv.InRange(imgHSV, imgThresholded,
		opencv.NewScalar(params.LowH, params.LowS, params.LowV, 0),
		opencv.NewScalar(params.HighH, params.HighS, params.HighV, 0))

	// Try to remove any noise
	opencv.Erode(imgThresholded, imgThresholded, opencv.CreateStructuringElement(5, 5, 2, 2, opencv.MORPH_ELLIPSE, nil), 1)
	opencv.Dilate(imgThresholded, imgThresholded, opencv.CreateStructuringElement(5, 5, 2, 2, opencv.MORPH_ELLIPSE, nil), 1)

	opencv.Dilate(imgThresholded, imgThresholded, opencv.CreateStructuringElement(5, 5, 2, 2, opencv.MORPH_ELLIPSE, nil), 1)
	opencv.Erode(imgThresholded, imgThresholded, opencv.CreateStructuringElement(5, 5, 2, 2, opencv.MORPH_ELLIPSE, nil), 1)

	// Try to find any large remaining blobs
	moments := opencv.Moments(imgThresholded, 0)
	if moments.M00() > 10000.0 {
		x := int(moments.M10() / moments.M00())
		y := int(moments.M01() / moments.M00())
		fmt.Printf("Found the target: %d, %d\n", x, y)
		return BallResult{true, x, y, imgThresholded}, nil
	} else {
		return BallResult{false, 0, 0, imgThresholded}, nil
	}
}

func main() {
	flag.Parse()
	calibrate := false
	if len(flag.Args()) > 0 && flag.Args()[0] == "calibrate" {
		calibrate = true
	}

	win := opencv.NewWindow("Go-OpenCV Webcam")
	defer win.Destroy()

	controlWin := opencv.NewWindow("Controls")
	defer controlWin.Destroy()

	// Setup the parameter window
	params, err := loadParams("config.json")
	if err != nil {
		fmt.Errorf("Error loading config: %s\n", err.Error())
	}
	controlWin.CreateTrackbar("LowH", int(params.LowH), 179, func(pos int, param ...interface{}) {
		params.LowH = float64(pos)
	})
	controlWin.CreateTrackbar("LowS", int(params.LowS), 255, func(pos int, param ...interface{}) {
		params.LowS = float64(pos)
	})
	controlWin.CreateTrackbar("LowV", int(params.LowV), 255, func(pos int, param ...interface{}) {
		params.LowV = float64(pos)
	})
	controlWin.CreateTrackbar("HighH", int(params.HighH), 179, func(pos int, param ...interface{}) {
		params.HighH = float64(pos)
	})
	controlWin.CreateTrackbar("HighS", int(params.HighS), 255, func(pos int, param ...interface{}) {
		params.HighS = float64(pos)
	})
	controlWin.CreateTrackbar("HighV", int(params.HighV), 255, func(pos int, param ...interface{}) {
		params.HighV = float64(pos)
	})

	// Setup the webcam
	cap := opencv.NewCameraCapture(0)
	if cap == nil {
		panic("can not open camera")
	}
	defer cap.Release()

	fmt.Println("Press ESC to quit")
	for {
		if cap.GrabFrame() {
			img := cap.RetrieveFrame(1)
			if img != nil {
				r, err := FindBall(img, win, params)
				if err != nil {
					fmt.Println("Error processing image")
					continue
				}
				if r.Found {
					// Draw a rectangle in the area of the target
					opencv.Rectangle(img,
						opencv.Point{r.X + 20, r.Y},
						opencv.Point{r.X, r.Y + 20},
						opencv.ScalarAll(255.0), 1, 1, 0)
				}
				if calibrate {
					params.Print()
					win.ShowImage(r.ProcessedImage)
				} else {
					win.ShowImage(img)
				}
			} else {
				fmt.Println("Image is nil")
			}
		}
		key := opencv.WaitKey(10)
		if key == 27 {
			writeParams("config.json", params)
			os.Exit(0)
		}
	}
}
