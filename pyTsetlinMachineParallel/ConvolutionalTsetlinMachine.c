/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "fast_rand.h"

#include "ConvolutionalTsetlinMachine.h"

struct TsetlinMachine *CreateTsetlinMachine(int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses)
{
	/* Set up the Tsetlin Machine structure */

	struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));

	tm->number_of_clauses = number_of_clauses;

	tm->number_of_features = number_of_features;

	tm->number_of_clause_chunks = (number_of_clauses-1)/32 + 1;

	tm->number_of_patches = number_of_patches;

	tm->clause_output = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clause_chunks);

	tm->output_one_patches = (int *)malloc(sizeof(int) * number_of_patches);

	tm->number_of_ta_chunks = number_of_ta_chunks;

	tm->feedback_to_la = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);

	tm->number_of_state_bits = number_of_state_bits;

	tm->ta_state = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses * number_of_ta_chunks * number_of_state_bits);

	tm->T = T;

	tm->s = s;

	tm->s_range = s_range;

	tm->clause_patch = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	tm->feedback_to_clauses = (int *)malloc(sizeof(int) * tm->number_of_clause_chunks);
	
	tm->clause_weights = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	if (((number_of_features) % 32) != 0) {
		tm->filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		tm->filter = 0xffffffff;
	}

	tm->boost_true_positive_feedback = boost_true_positive_feedback;

	tm->weighted_clauses = weighted_clauses;
	
	tm_initialize(tm);

	return tm;
}

void tm_initialize(struct TsetlinMachine *tm)
{
	/* Set up the Tsetlin Machine structure */

	unsigned int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits-1; ++b) {											
				tm->ta_state[pos] = ~0;
				pos++;
			}
			tm->ta_state[pos] = 0;
			pos++;
		}
		tm->clause_weights[j] = 1;
	}
}

void tm_destroy(struct TsetlinMachine *tm)
{
	free(tm->clause_output);
	free(tm->output_one_patches);
	free(tm->feedback_to_la);
	free(tm->ta_state);
	free(tm->feedback_to_clauses);
	free(tm->clause_weights);
}

static inline void tm_initialize_random_streams(struct TsetlinMachine *tm, int clause)
{
	// Initialize all bits to zero	
	memset(tm->feedback_to_la, 0, tm->number_of_ta_chunks*sizeof(unsigned int));

	int n = tm->number_of_features;
	double p = 1.0 / (tm->s + 1.0 * clause * (tm->s_range - tm->s) / tm->number_of_clauses);

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (tm->number_of_features);
		while (tm->feedback_to_la[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (tm->number_of_features);
	    }
		tm->feedback_to_la[f / 32] |= 1 << (f % 32);
	}
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_inc(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (carry == 0)
			break;

		carry_next = ta_state[b] & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] |= carry;
		}
	} 	
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_dec(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (carry == 0)
			break;

		carry_next = (~ta_state[b]) & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] &= ~carry;
		}
	} 
}

/* Sum up the votes for each class 
If clamp is 1, then the class sum is clamped between -T and T
This should always be true unless you are crazy (or using soft labels)
*/
static inline int sum_up_class_votes(struct TsetlinMachine *tm, int clamp)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		if (j % 2 == 0) {
			class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		} else {
			class_sum -= tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		}	
	}

	if (clamp == 1) {
		class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;
		class_sum = (class_sum < -(tm->T)) ? -(tm->T) : class_sum;
	}

	return class_sum;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void tm_calculate_clause_output(struct TsetlinMachine *tm, unsigned int *Xi, int predict)
{
	int output_one_patches_count;

	unsigned int *ta_state = tm->ta_state;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		output_one_patches_count = 0;

		for (int patch = 0; patch < tm->number_of_patches; ++patch) {
			unsigned int output = 1;
			unsigned int all_exclude = 1;
			for (int k = 0; k < tm->number_of_ta_chunks-1; k++) {
				unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;
				output = output && (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos];

				if (!output) {
					break;
				}
				all_exclude = all_exclude && (ta_state[pos] == 0);
			}

			unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + (tm->number_of_ta_chunks-1)*tm->number_of_state_bits + tm->number_of_state_bits-1;
			output = output &&
				(ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + tm->number_of_ta_chunks - 1] & tm->filter) ==
				(ta_state[pos] & tm->filter);

			all_exclude = all_exclude && ((ta_state[pos] & tm->filter) == 0);

			output = output && !(predict == PREDICT && all_exclude == 1);

			if (output) {
				tm->output_one_patches[output_one_patches_count] = patch;
				output_one_patches_count++;
			}
		}
	
		if (output_one_patches_count > 0) {
			unsigned int clause_chunk = j / 32;
			unsigned int clause_chunk_pos = j % 32;

 			tm->clause_output[clause_chunk] |= (1 << clause_chunk_pos);

 			int patch_id = fast_rand() % output_one_patches_count;
	 		tm->clause_patch[j] = tm->output_one_patches[patch_id];
 		}
 	}
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_clauses(struct TsetlinMachine *tm, unsigned int *Xi, int class_sum, int target)
{
	unsigned int *ta_state = tm->ta_state;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		float T_scale = 2;
	 	tm->feedback_to_clauses[clause_chunk] |= 
		    (((float)fast_rand())/((float)FAST_RAND_MAX) <= 
		    (1.0/(tm->T*T_scale))*(tm->T + (1 - 2*target)*class_sum)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}

		omp_set_lock(&tm->clause_lock[j]);	
		if ((2*target-1) * (1 - 2 * (j & 1)) == -1) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
			}
		} else if ((2*target-1) * (1 - 2 * (j & 1)) == 1) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type Ia Feedback

				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
			} else {
				// Type Ib Feedback
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
		omp_unset_lock(&tm->clause_lock[j]);
	}
}

void tm_update(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/


	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes(tm, 1);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/

	tm_update_clauses(tm, Xi, class_sum, target);
}

/*
If clamp is 1, then the class sum is clamped between -T and T
This should always be true unless you are crazy (or using soft labels)
*/
int tm_score(struct TsetlinMachine *tm, unsigned int *Xi, int clamp) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes(tm, clamp);
}

int tm_ta_state(struct TsetlinMachine *tm, int clause, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits;

	int state = 0;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (tm->ta_state[pos + b] & (1 << chunk_pos)) {
			state |= 1 << b; 
		}
	}

	return state;
}

int tm_ta_action(struct TsetlinMachine *tm, int clause, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits + tm->number_of_state_bits-1;

	return (tm->ta_state[pos] & (1 << chunk_pos)) > 0;
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void tm_get_ta_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				ta_state[pos] = tm->ta_state[pos];
				pos++;
			}
		}
	}
}

void tm_set_ta_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				tm->ta_state[pos] = ta_state[pos];
				pos++;
			}
		}
	}
}

void tm_get_clause_weights(struct TsetlinMachine *tm, unsigned int *clause_weights)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		clause_weights[j] = tm->clause_weights[j];
	}
}

void tm_set_clause_weights(struct TsetlinMachine *tm, unsigned int *clause_weights)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		tm->clause_weights[j] = clause_weights[j];
	}
}

/**************************************/
/*** The Regression Tsetlin Machine ***/
/**************************************/

/* Sum up the votes for each class */
static inline int sum_up_class_votes_regression(struct TsetlinMachine *tm)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		
	}
	class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;

	return class_sum;
}

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
	unsigned int *ta_state = tm->ta_state;

	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes_regression(tm);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/
	
	// Calculate feedback to clauses

	int prediction_error = class_sum - target; 

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

	 	tm->feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)FAST_RAND_MAX) <= pow(1.0*prediction_error/tm->T, 2)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}
		
		if (prediction_error > 0) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
			}
		} else if (prediction_error < 0) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type Ia Feedback
				
				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
			} else {
				// Type Ib Feedback
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

void tm_fit_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	int max_threads = omp_get_max_threads();
	struct TsetlinMachine **tm_thread = (void *)malloc(sizeof(struct TsetlinMachine *) * max_threads);

	tm->clause_lock = (omp_lock_t *)malloc(sizeof(omp_lock_t) * tm->number_of_clauses);
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		omp_init_lock(&tm->clause_lock[j]);
	}
	
	for (int t = 0; t < max_threads; t++) {
		tm_thread[t] = CreateTsetlinMachine(tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		free(tm_thread[t]->ta_state);
		tm_thread[t]->ta_state = tm->ta_state;
		free(tm_thread[t]->clause_weights);
		tm_thread[t]->clause_weights = tm->clause_weights;
		tm_thread[t]->clause_lock = tm->clause_lock;
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...

		#pragma omp parallel for
		for (int l = 0; l < number_of_examples; l++) {
			int thread_id = omp_get_thread_num();
			unsigned int pos = l*step_size;

			tm_update_regression(tm_thread[thread_id], &X[pos], y[l]);
		}
	}

	for (int t = 0; t < max_threads; t++) {
		free(tm_thread[t]->clause_output);
		free(tm_thread[t]->output_one_patches);
		free(tm_thread[t]->feedback_to_la);
		free(tm_thread[t]->feedback_to_clauses);
		free(tm_thread[t]);
	}

	for (int j = 0; j < tm->number_of_clauses; ++j) {
		omp_destroy_lock(&tm->clause_lock[j]);
	}

	free(tm->clause_lock);
	free(tm_thread);
}

int tm_score_regression(struct TsetlinMachine *tm, unsigned int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes_regression(tm);
}

/******************************/
/*** Predict y for inputs X ***/
/******************************/

void tm_predict_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	int max_threads = omp_get_max_threads();
	struct TsetlinMachine **tm_thread = (void *)malloc(sizeof(struct TsetlinMachine *) * max_threads);
	
	for (int t = 0; t < max_threads; t++) {
		tm_thread[t] = CreateTsetlinMachine(tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		free(tm_thread[t]->ta_state);
		tm_thread[t]->ta_state = tm->ta_state;
		free(tm_thread[t]->clause_weights);
		tm_thread[t]->clause_weights = tm->clause_weights;
	}

	#pragma omp parallel for
	for (int l = 0; l < number_of_examples; l++) {
		int thread_id = omp_get_thread_num();
		unsigned int pos = l*step_size;

		y[l] = tm_score_regression(tm_thread[thread_id], &X[pos]);		
	}

	for (int t = 0; t < max_threads; t++) {
		free(tm_thread[t]->clause_output);
		free(tm_thread[t]->output_one_patches);
		free(tm_thread[t]->feedback_to_la);
		free(tm_thread[t]->feedback_to_clauses);
		free(tm_thread[t]);
	}

	free(tm_thread);
	
	return;
}
