#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/*****************************************************************************
** DATA STRUCTURES
*****************************************************************************/

typedef struct SAMPLE{
	int label;
	int word_count;
	int *start_word;
} sample;

typedef struct DATASET{
	long word_count;
	char **words;
	long sample_count;
	sample *samples;
	int *buf;
	char *word_buf;
} dataset;

typedef struct NN_WEIGHTS{
	int word_count;
	int hidden_units;
	double *weights_i_h;
	double *bias_i_h;
	double *weights_h_o;
	double *bias_h_o;
} nn_weights;

typedef struct TRAIN_THREAD_SUBSET{
	nn_weights *W;
	nn_weights *delta;
	dataset *data;
	int start;
	int end;
	int subsample;
} train_thread_subset;

/*****************************************************************************
** SORTING COMPARATORS 
*****************************************************************************/
int cmpint(const void* a, const void* b)
{
    int aa = *(int *)a;
    int bb = *(int *)b;
    return aa-bb;
}
int cmpstr(const void* a, const void* b)
{
    const char* aa = *(const char**)a;
    const char* bb = *(const char**)b;
    return strcmp(aa,bb);
}

/*****************************************************************************
** DATASET LOADING
**
** CAUTION: all file read to memory 
*****************************************************************************/
int load_dataset(FILE *fp, dataset *res, char **train_words, int train_word_count, int cutoff){
	// loads a dataset file. file must contain lines starting with -1,0 or 1
	// folowed by the words separated by space
	//
	// params:
	//	fp: file pointer to a dataset file
	//	res: pointer to the dataset structure storing the result
	//	train_words: pointer to the dictionary (sorted array of words)
	//	train_word_count: the size of the dictionary
	//	cutoff: words with less than cutoff occurances are skipped from dictionary
	// returns:
	//	0 for success or error code
	
	long size;
	char *buf;
	char *pch;
	char prch;
	long line_count;
	long word_count;
	long unique_word_count;
	int *labels;
	char **line_starts;
	char **line_ends;
	char **words;
	int lc;
	int wc;
	int w;
	int bck;
	int cw_bck;
	int lwc;
	int *inputs = NULL;
	int inputs_capacity = 0;
	int i,j;
	// get file size
	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	// read the whole file in memory
	fseek(fp, 0, SEEK_SET);
	buf = (char *)malloc(size+1);
	fread(buf,1, size, fp);
	buf[size] = 0;
	// get basic stats
	pch = buf;
	prch = 0;
	line_count = 0;
	word_count = 0;
	while(*(pch++)){
		if(*pch == '\n' || (*pch == 0 && prch != '\n')){
			line_count++;
			word_count++;
		}else if(*pch == ' '){
			word_count++;
		}
		prch = *pch;
	}
	// remove labels
	word_count -= line_count;
	// parse
	labels = (int *)malloc(sizeof(int)*line_count);
	line_starts = (char **)malloc(sizeof(char *)*(line_count+1));
	line_ends = (char **)malloc(sizeof(char *)*line_count);
	words = (char **)malloc(sizeof(char *)*(word_count+1));
	lc = 0;
	pch = buf;
	if(*pch == '0'){ 
		labels[lc] = 0;
		if(*(++pch) != ' '){
			fprintf(stderr, "expecting space after 0 at line %d\n", lc);
			return 1;
		}
		pch++;
	}else if (*pch == '1'){
		labels[lc] = 1;
		if(*(++pch) != ' '){
			fprintf(stderr, "expecting space after 1 at line %d\n", lc);
			return 1;
		}
		pch++;
	}else if (*pch == '-'){
		labels[lc] = -1;
		if(*(++pch) != '1'){
			fprintf(stderr, "expecting 1 after - at line %d\n", lc);
			return 1;
		}
		if(*(++pch) != ' '){
			fprintf(stderr, "expecting space after -1 at line %d\n", lc);
			return 1;
		}
		pch++;
	}else{
		fprintf(stderr, "line %d should start with 0, 1 or -1\n", lc);
		return 1;
	}
	line_starts[0] = pch;
	words[0] = pch;
	wc = 1;
	while(*pch){
		if(*pch == '\n' || !*pch){
			line_ends[lc++] = pch;
			if(!*pch) break;
			*(pch++) = 0;
			if(!*pch) break;
			if(*pch == '0'){ 
				labels[lc] = 0;
				if(*(++pch) != ' '){
					fprintf(stderr, "expecting space after 0 at line %d\n", lc);
					return 1;
				}
				pch++;
			}else if (*pch == '1'){
				labels[lc] = 1;
				if(*(++pch) != ' '){
					fprintf(stderr, "expecting space after 1 at line %d\n", lc);
					return 1;
				}
				pch++;
			}else if (*pch == '-'){
				labels[lc] = -1;
				if(*(++pch) != '1'){
					fprintf(stderr, "expecting 1 after - at line %d\n", lc);
					return 1;
				}
				if(*(++pch) != ' '){
					fprintf(stderr, "expecting space after -1 at line %d\n", lc);
					return 1;
				}
				pch++;
			}else{
				fprintf(stderr, "line %d should start with 0, 1 or -1\n", lc);
				return 1;
			}
			line_starts[lc] = pch;
			words[wc++] = pch;
		}else if(*pch == ' '){
			*(pch++) = 0;
			words[wc++] = pch;
		}else{
			pch++;
		}
	}
	if(lc != line_count){
		fprintf(stderr, "internal error 1 %d!=%d\n", lc, line_count);
		return 1;
	}
	if(wc != word_count){
		fprintf(stderr, "internal error 2 %d!=%d\n", wc, word_count);
		return 1;
	}
	// make a unique list of words
	qsort(words, word_count, sizeof(char *), cmpstr);
	bck = 0;
	cw_bck = 0;
	for(w = 1; w  < word_count; w++){
		if(strcmp(words[w-bck-1], words[w]) == 0){
			bck++;
			cw_bck++;
		}else{
			if(cw_bck < cutoff) bck++;
			words[w-bck] = words[w];
			cw_bck = 0;
		}
	}
	unique_word_count = word_count - bck;
	// get train words
	if(!train_words){
		train_words = words;
		train_word_count = unique_word_count;
	}
	// prepare samples
	res->sample_count = line_count;
	res->samples = (sample *)malloc(sizeof(sample)*res->sample_count);
	res->buf = (int *)malloc(sizeof(int)*word_count);
	res->word_count = unique_word_count + 1;
	// set words
	res->words = words;
	size = 0;
	for(i = 0; i < word_count; i++){
		size += strlen(words[i])+1;
	}
	res->word_buf = (char *)malloc(sizeof(char) * size);
	size = 0;
	for(i = 0; i < word_count; i++){
		j = strlen(words[i])+1;
		memcpy(res->word_buf + size, words[i], j);
		res->words[i] = res->word_buf + size;
		size += j;
	}
	// loop through lines in data
	wc = 0;
	i = 0;
	for(lc = 0; lc < line_count; lc++){
		res->samples[lc].label = labels[lc];
		res->samples[lc].word_count = 0;
		res->samples[lc].start_word = res->buf + i;
		// skip the label
		while(*(pch++)!=' ');
		// get word indeces
		lwc = 0;
		pch = line_starts[lc];
		while(1){
			wc++;
			char **pword = bsearch(&pch, train_words, train_word_count, sizeof(char *), cmpstr);
			int word_idx = (pword) ? pword - train_words + 1 : 0;
			lwc++;
			if(lwc >= inputs_capacity){
				inputs_capacity = (inputs_capacity==0) ? 1000 : 2 * inputs_capacity;
				inputs = (int *)realloc(inputs, sizeof(int)*inputs_capacity);
			}
			inputs[lwc - 1] = word_idx;
			while(*pch && pch != line_ends[lc]) pch++;
			if(pch == line_ends[lc]) break;
			pch++;
		}
		// sort unique word indices
		qsort(inputs, lwc, sizeof(int), cmpint);
		bck = 0;
		for(j = 1; j < lwc; j++){
			if(inputs[j-bck-1] == inputs[j]) bck++;
			else inputs[j-bck] = inputs[j];
		}
		// update samples and buffer
		res->samples[lc].word_count = lwc - bck;
		for(j = 0; j < res->samples[lc].word_count; j++){
			res->buf[i++] = inputs[j];
		}
	}
	if(lc != line_count){
		fprintf(stderr, "internal error 1 %d!=%d\n", lc, line_count);
		return 1;
	}
	if(wc != word_count){
		fprintf(stderr, "internal error 2 %d!=%d\n", wc, word_count);
		return 1;
	}
	free(buf);
	free(line_starts);
	free(line_ends);
	free(labels);
	return 0;
}

/*****************************************************************************
** WEIGHTS  
*****************************************************************************/
void weights_init(int word_count, int hidden_units, nn_weights *W, int init_random){
	int i,j;
	word_count++;
	W->word_count = word_count;
	W->hidden_units = hidden_units;
	W->weights_i_h = (double *)malloc(sizeof(double)*word_count*hidden_units);
	W->bias_i_h = (double *)malloc(sizeof(double)*hidden_units);
	W->weights_h_o = (double *)malloc(sizeof(double)*hidden_units);
	W->bias_h_o = (double *)malloc(sizeof(double));
	for(i=word_count*hidden_units-1;i>=0;i--) W->weights_i_h[i] = (init_random) ? ((double)rand()) / ((unsigned)RAND_MAX + 1) - 0.5 : 0;
	for(i=hidden_units-1;i>=0;i--) W->bias_i_h[i] = (init_random) ? ((double)rand()) / ((unsigned)RAND_MAX + 1) - 0.5 : 0;
	for(i=hidden_units-1;i>=0;i--) W->weights_h_o[i] = (init_random) ? ((double)rand()) / ((unsigned)RAND_MAX + 1) - 0.5 : 0;
	W->bias_h_o[0] = (init_random) ? ((double)rand()) / ((unsigned)RAND_MAX + 1) - 0.5 : 0;
}

void weights_save(nn_weights *W, const char *filename){
	FILE *fp = fopen(filename, "wb");
	fwrite(&W->word_count, sizeof(W->word_count), 1, fp);
	fwrite(&W->hidden_units, sizeof(W->hidden_units), 1, fp);
	fwrite(W->weights_i_h, sizeof(double), W->word_count * W->hidden_units, fp);
	fwrite(W->bias_i_h, sizeof(double), W->hidden_units, fp);
	fwrite(W->weights_h_o, sizeof(double), W->hidden_units, fp);
	fwrite(W->bias_h_o, sizeof(double), 1, fp);
	fclose(fp);
}

void weights_load(nn_weights *W, const char *filename){
	FILE *fp = fopen(filename, "rb");
	fread(&W->word_count, sizeof(W->word_count), 1, fp);
	fread(&W->hidden_units, sizeof(W->hidden_units), 1, fp);
	W->weights_i_h = (double *)malloc(sizeof(double)*W->word_count*W->hidden_units);
	W->bias_i_h = (double *)malloc(sizeof(double)*W->hidden_units);
	W->weights_h_o = (double *)malloc(sizeof(double)*W->hidden_units);
	W->bias_h_o = (double *)malloc(sizeof(double));
	fread(W->weights_i_h, sizeof(double), W->word_count * W->hidden_units, fp);
	fread(W->bias_i_h, sizeof(double), W->hidden_units, fp);
	fread(W->weights_h_o, sizeof(double), W->hidden_units, fp);
	fread(W->bias_h_o, sizeof(double), 1, fp);
	fclose(fp);
}
/*****************************************************************************
** ACTIVATIONS 
*****************************************************************************/
double activation_sigmoid(double x){
	return 1/(1+exp(-x));
}

double activation_sigmoid_derivative(double x){
	double s = activation_sigmoid(x);
	return s * (1 - s);
}

double activation_tanh(double x){
	double ep = exp(x);
	double em = exp(-x);
	return (ep-em)/(ep+em);
}

double activation_tanh_derivative(double x){
	double s = activation_tanh(x);
	return 1 - s * s;
}

double activation_relu(double x){
	return (x>0) ? x : 0;
}

double activation_relu_derivative(double x){
	return (x>0) ? 1 : 0;
}

double activation_hidden(double x){
	return activation_tanh(x);
}

double activation_hidden_derivative(double x){
	return activation_tanh_derivative(x);
}

double activation_measure(double x){
	return activation_sigmoid(x);
}

double activation_measure_derivative(double x){
	return activation_sigmoid_derivative(x);
}

double activation_score(double x){
	return activation_tanh(x);
}

double activation_score_derivative(double x){
	return activation_tanh_derivative(x);
}


/*****************************************************************************
** TESTING / FEED FORWARD
*****************************************************************************/
double test(nn_weights *W, dataset *data, double *mae, int print_confusion_matrix){
	double *hidden_outputs = (double *)malloc(W->hidden_units * sizeof(double));
	int *inputs = NULL;
	double output_measure;
	double output_score;
	int i,j,w, wc;
	int cm[3][3];
	int a;
	int p;
	// initialize mae 
	*mae = 0;
	// initialize confusion matrix
	for(a = 0; a < 3; a++){
		for(p = 0; p < 3; p++){
			cm[a][p] = 0;
		}
	}
	// loop through lines in data
	for(i = 0; i < data->sample_count; i++){
		wc = data->samples[i].word_count;
		inputs = data->samples[i].start_word; 
		// Forward pass: get hidden outputs
		memset(hidden_outputs, 0, sizeof(double)*W->hidden_units);
		for(j = 0; j < W->hidden_units; j++){
			for(w = 0; w < wc; w++){
				hidden_outputs[j] += W->weights_i_h[j*W->word_count+inputs[w]];
			}
			hidden_outputs[j] += W->bias_i_h[j];
			hidden_outputs[j] = activation_hidden(hidden_outputs[j]);
		}
		// Forward pass: get measure
		output_measure = 0;
		for(j = 0; j < W->hidden_units; j++){
			output_measure += W->weights_h_o[j] * hidden_outputs[j];
		}
		output_measure += W->bias_h_o[0];
		output_measure = activation_measure(output_measure);
		// update confusion matrix
		a = abs(data->samples[i].label);
		p = (output_measure < 0.5) ? 0 : 1; // ((output_score > 0)? 2 : 0);
		cm[a][p]++;
		// update mae
		*mae += fabs(a-output_measure);// + fabs(a-p);
	}
	*mae /= data->sample_count;
	// print confusion matrix
	int total = 0;
	int correct = 0;
	for(a = 0; a < 2; a++){
		if(print_confusion_matrix) printf("|\t");
		for(p = 0; p < 2; p++){
			if(print_confusion_matrix) printf("% 6d\t|\t",cm[a][p]);
			total += cm[a][p];
			if(a==p) correct += cm[a][p];
		}
		if(print_confusion_matrix) printf("\n");
	}
	free(hidden_outputs);
	return 100. * correct / total;
}

/*****************************************************************************
** TRAINING / FEED FORWARD & BACKPROPAGATION
*****************************************************************************/
void *thread_train(void *vargp){
	train_thread_subset *prms = (train_thread_subset *)vargp;
	dataset *data = prms->data;
	nn_weights *W = prms->W;
	double *wp;
	int *inputs = NULL;
	double output_measure;
	double output_score;
	double *hidden_outputs = NULL;
	hidden_outputs = (double *)malloc(W->hidden_units * sizeof(double));
	int i,j;
	int w, wc, jw;
	double error_measure_sum = 0;
	// loop through lines in data
	for(i = prms->start; i < prms->end; i++){
		if(rand() > RAND_MAX / prms->subsample) continue;
		wc = data->samples[i].word_count;
		inputs = data->samples[i].start_word; 
		// Forward pass: get hidden outputs
		memset(hidden_outputs, 0, sizeof(double)*W->hidden_units);
		for(j = 0; j < W->hidden_units; j++){
			for(w = 0, wp = W->weights_i_h + j * W->word_count; w < wc; w++){
				if(inputs[w] < 0 || inputs[w] >= W->word_count){
				}
				hidden_outputs[j] += wp[inputs[w]];
			}
			hidden_outputs[j] += W->bias_i_h[j];
			hidden_outputs[j] = activation_hidden(hidden_outputs[j]);
		}
		// Forward pass: get measure
		output_measure = 0;
		for(j = 0, wp = W->weights_h_o; j < W->hidden_units; j++, wp++){
			output_measure += *wp * hidden_outputs[j];
		}
		output_measure += W->bias_h_o[0];
		output_measure = activation_measure(output_measure);
		// calculate errors
		double output_error_measure = abs(data->samples[i].label) - output_measure;
		error_measure_sum += fabs(output_error_measure);// + fabs(output_error_score);
		double output_error_term_measure = output_error_measure * activation_measure_derivative(output_measure);
		for(j = 0; j < W->hidden_units; j++){
			double hidden_error_term = W->weights_h_o[j] * output_error_term_measure * activation_hidden_derivative(hidden_outputs[j]);
			for(w = 0; w < wc; w++){
				prms->delta->weights_i_h[W->word_count*j + inputs[w]] += hidden_error_term * 1;
			}
			prms->delta->bias_i_h[j] += hidden_error_term;
		}
		for(j = 0; j < W->hidden_units; j++){
			prms->delta->weights_h_o[j] += output_error_term_measure * hidden_outputs[j];
		}
		prms->delta->bias_h_o[0] += output_error_term_measure;
	}
	free(hidden_outputs);
	return NULL;
}

void train_epoch(int threads, int subsample, double lr, nn_weights *W, dataset *data){
	int j, w, s;
	train_thread_subset *thread_data = (train_thread_subset *)malloc(sizeof(train_thread_subset) * threads);
	pthread_t *thread_ids = (pthread_t *)malloc(sizeof(pthread_t) * threads);
	nn_weights delta;
	weights_init(W->word_count, W->hidden_units, &delta, 0); 
	for(s = 0; s < subsample; s++){
		for(j = 0; j < threads; j++){
			thread_data[j].subsample = subsample;
			thread_data[j].W = W;
			thread_data[j].delta = &delta;
			thread_data[j].data = data;
			thread_data[j].start = j * data->sample_count / threads;
			thread_data[j].end = (j+1) * data->sample_count / threads;
			pthread_create(thread_ids+j, NULL, thread_train, thread_data+j);
		}
		for(j = 0; j < threads; j++){
			pthread_join(thread_ids[j], NULL);
		}
		for(j = 0; j < W->hidden_units; j++){
			for(w = 0; w < W->word_count; w++){
				W->weights_i_h[W->hidden_units*j + w] += lr * delta.weights_i_h[W->hidden_units*j + w] / data->sample_count;
			}
			W->bias_i_h[j] += lr * delta.bias_i_h[j] / data->sample_count;
		}
		for(j = 0; j < W->hidden_units; j++){
			W->weights_h_o[j] += lr * delta.weights_h_o[j] / data->sample_count;
		}
		W->bias_h_o[0] += lr * delta.bias_h_o[0] / data->sample_count;
	}
	free(delta.weights_i_h);
	free(delta.bias_i_h);
	free(delta.weights_h_o);
	free(delta.bias_h_o);
	free(thread_ids);
	free(thread_data);
}

int main(int argc, const char **argv){
	if(argc <= 3){
		printf("please specify a training and validation file name\n");
		return 1;
	}
	FILE *train_fp = fopen(argv[1],"r");
	FILE *valid_fp = fopen(argv[2],"r");
	FILE *test_fp = fopen(argv[3],"r");
	int cutoff = 2;
	int decay_step = 100;
	int hidden_units = 50;
	int subsample = 10;
	int t = 8; // NUMBER_OF_THREADS
	if(!train_fp){
		printf("could not find %s\n", argv[1]);
		return 1;
	}
	if(!valid_fp){
		printf("could not find %s\n", argv[2]);
		return 1;
	}
	if(!valid_fp){
		printf("could not find %s\n", argv[3]);
		return 1;
	}
	dataset train_set;
	int rv = load_dataset(train_fp, &train_set, NULL, -1, cutoff);
	if(rv != 0){
		return rv;
	}
	fclose(train_fp);
	printf("TRAIN: %d samples\n", train_set.sample_count);
	printf("TRAIN: %d words\n", train_set.word_count);
	dataset valid_set;
	rv = load_dataset(valid_fp, &valid_set, train_set.words, train_set.word_count, cutoff);
	if(rv != 0){
		return rv;
	}
	fclose(valid_fp);
	printf("VALID: %d samples\n", valid_set.sample_count);
	printf("VALID: %d words\n", valid_set.word_count);
	dataset test_set;
	rv = load_dataset(test_fp, &test_set, train_set.words, train_set.word_count, cutoff);
	if(rv != 0){
		return rv;
	}
	fclose(test_fp);
	printf("TEST: %d samples\n", test_set.sample_count);
	printf("TEST: %d words\n", test_set.word_count);
	nn_weights W;
	if(1){
		weights_init(train_set.word_count, hidden_units, &W, 1); 
	}else{
		weights_load(&W,"nn.weights");
	}
	printf("Inited weights...\n");
	int epoch;
	double lr=1;
	double best_acc = 0;
	double best_vacc = 0;
	double best_vmae = 0;
	int grace = 0;
	double best_tacc = 0;
	double mae = 0;
	double best_lr = 0;
	double lr_cand = 0;
	double best_mae = 0;
	int e;
	// lr_range_test
	printf("learning rate range test...\n");
	weights_save(&W, "nn.weights");
	for(lr = 0.01; lr <= 2; lr /= 0.9){
		train_epoch(t, subsample, lr, &W, &train_set);
		test(&W, &train_set, &mae, 0);
		if(best_lr == 0 || best_mae > mae){
			best_mae = mae;
			grace++;
			if(grace == 1) lr_cand = lr/2;
			if(best_lr == 0 || grace == 3){
				best_lr = lr_cand;
				grace = 0;
			}
		}else{
			grace--;
			if(grace == -3) break;
		}
	}
	lr = best_lr;
	grace = 0;
	printf("learning rate = %lf\n", lr);
	weights_load(&W, "nn.weights");
	// training loop 
	for(epoch = 0; epoch < 10000; epoch++){
		train_epoch(t, subsample, lr, &W, &train_set);
		if(epoch % 20 == 0){
			printf("-------------------------------------------------\n");
			if(grace > 0){
				printf("GRACE %d!!!!\n", grace);
			}
			printf("epoch: %d, learning rate: %.2lf\n", epoch, lr);
			double acc = test(&W, &train_set, &mae, 1);
			if(best_acc < acc) best_acc = acc;
			printf("best_acc: %.2lf, acc: %.2lf, mae: %lf\n", best_acc, acc, mae);
			double vacc = test(&W, &valid_set, &mae, 1);
			if(best_vacc < vacc){
				best_vacc = vacc;
				printf("Saving weights...\n");
				weights_save(&W, "nn.weights");
			}
			printf("best_vacc: %.2lf, vacc: %.2lf, mae: %lf\n", best_vacc, vacc, mae);
			if(epoch == 0 || mae < best_vmae){
				best_vmae = mae;
				grace = 0;
			}else{
				grace++;
				if(lr <= 0.01) break;
				lr *= 0.9;
				weights_load(&W, "nn.weights");
			}
			double tacc = test(&W, &test_set, &mae, 1);
			if(best_tacc < tacc) best_tacc = tacc;
			printf("best_tacc: %.2lf, tacc: %.2lf, mae: %lf\n", best_tacc, tacc, mae);
			if(grace > 5){
				break;
			}
		}
		if(lr > 0.01 && epoch % decay_step == decay_step-1) lr *= 0.9;
	}
	printf("==========================================\n");
	weights_load(&W, "nn.weights");
	double acc = test(&W, &train_set, &mae, 1);
	if(best_acc < acc) best_acc = acc;
	printf("best_acc: %.2lf, acc: %.2lf, mae: %lf\n", best_acc, acc, mae);
	if(epoch % decay_step == decay_step-1) lr *= 0.9;
	double vacc = test(&W, &valid_set, &mae, 1);
	if(best_vacc < vacc) best_vacc = vacc;
	printf("best_vacc: %.2lf, vacc: %.2lf, mae: %lf\n", best_vacc, vacc, mae);
	double tacc = test(&W, &test_set, &mae, 1);
	if(best_tacc < tacc) best_tacc = tacc;
	printf("best_tacc: %.2lf, tacc: %.2lf, mae: %lf\n", best_tacc, tacc, mae);
	return 0;
}
