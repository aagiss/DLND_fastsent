all:
	gcc -O3 -o fastsent_train train.c -lm -lpthread
