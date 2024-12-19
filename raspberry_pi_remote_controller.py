import RPi.GPIO as GPIO
from time import sleep

FORWARD_PIN = 27
BACK_PIN = 22
RIGHT_PIN = 5
LEFT_PIN = 6

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(FORWARD_PIN, GPIO.OUT)
GPIO.setup(BACK_PIN, GPIO.OUT)
GPIO.setup(RIGHT_PIN, GPIO.OUT)
GPIO.setup(LEFT_PIN, GPIO.OUT)

GPIO.output(FORWARD_PIN, GPIO.LOW)
GPIO.output(BACK_PIN, GPIO.LOW)
GPIO.output(RIGHT_PIN, GPIO.LOW)
GPIO.output(LEFT_PIN, GPIO.LOW)

def forward():
	GPIO.output(BACK_PIN, GPIO.LOW)
	GPIO.output(FORWARD_PIN, GPIO.HIGH)

def back():
    GPIO.output(FORWARD_PIN, GPIO.LOW)
    GPIO.output(BACK_PIN, GPIO.HIGH)

def right():
    GPIO.output(LEFT_PIN, GPIO.LOW)
    GPIO.output(RIGHT_PIN, GPIO.HIGH)

def left():
    GPIO.output(RIGHT_PIN, GPIO.LOW)
    GPIO.output(LEFT_PIN, GPIO.HIGH)

def stop():
    GPIO.output(FORWARD_PIN, GPIO.LOW)
    GPIO.output(BACK_PIN, GPIO.LOW)
    GPIO.output(RIGHT_PIN, GPIO.LOW)
    GPIO.output(LEFT_PIN, GPIO.LOW)

def turn_left(angle):
	back()
	sleep(0.125)
	for x in range(round(angle/7.5)):
		forward()
		left()
		sleep(0.125)
		back()
		right()
		sleep(0.145)
	stop()
	sleep(0.1)

def turn_right(angle):
	back()
	sleep(0.125)
	for x in range(round(angle/7.5)):
		forward()
		right()
		sleep(0.125)
		back()
		left()
		sleep(0.145)
	stop()
	sleep(0.1)

def go_forward(distance):
	for x in range(round(distance/5.25)):
		forward()
		sleep(0.1)
		back()
		sleep(0.075)
		stop()
		sleep(0.1)
	stop()

def go_backward(distance):
	for x in range(round(distance/3.8)):
		back()
		sleep(0.1)
		forward()
		sleep(0.06)
		stop()
		sleep(0.1)
	stop()
