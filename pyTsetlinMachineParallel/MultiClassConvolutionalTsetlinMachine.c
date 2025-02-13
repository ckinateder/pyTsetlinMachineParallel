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
#include <omp.h>

#include "MultiClassConvolutionalTsetlinMachine.h"

/**************************************/
/*** The Convolutional Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses)
{

	struct MultiClassTsetlinMachine *mc_tm = NULL;

	mc_tm = (void *)malloc(sizeof(struct MultiClassTsetlinMachine));

	mc_tm->number_of_classes = number_of_classes;
	mc_tm->tsetlin_machines = (void *)malloc(sizeof(struct TsetlinMachine *)* number_of_classes);
	for (int i = 0; i < number_of_classes; i++) {
		mc_tm->tsetlin_machines[i] = CreateTsetlinMachine(number_of_clauses, number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, s_range, boost_true_positive_feedback, weighted_clauses);
	}
	
	mc_tm->number_of_patches = number_of_patches;

	mc_tm->number_of_ta_chunks = number_of_ta_chunks;

	mc_tm->number_of_state_bits = number_of_state_bits;

	return mc_tm;
}

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_initialize(mc_tm->tsetlin_machines[i]);
	}
}

void mc_tm_destroy(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_destroy(mc_tm->tsetlin_machines[i]);

		free(mc_tm->tsetlin_machines[i]);
	}
	free(mc_tm->tsetlin_machines);
}

/***********************************/
/*** Predict classes of inputs X ***/
/***********************************/

void mc_tm_predict(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples)
{

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		for (int i = 0; i < mc_tm->number_of_classes; i++) {
			free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
			mc_tm_thread[t]->tsetlin_machines[i]->ta_state = mc_tm->tsetlin_machines[i]->ta_state;
			free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
			mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = mc_tm->tsetlin_machines[i]->clause_weights;
		}	
	}

	#pragma omp parallel for
	for (int l = 0; l < number_of_examples; l++) {
		int thread_id = omp_get_thread_num();

		unsigned int pos = l*step_size;
		// Identify class with largest output
		int max_class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[0], &X[pos]);
		int max_class = 0;
		for (int i = 0; i < mc_tm_thread[thread_id]->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		y[l] = max_class;
	}

	for (int t = 0; t < max_threads; t++) {
		for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {	

			struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];

			free(tm_thread->clause_output);
			free(tm_thread->output_one_patches);
			free(tm_thread->feedback_to_la);
			free(tm_thread->feedback_to_clauses);
			free(tm_thread);
		}
	}

	free(mc_tm_thread);
	
	return;
}

/***********************************/
/*** Predict classes of inputs X ***/
/*** and return the max class sums */
/***********************************/

void mc_tm_predict_with_class_sums(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int *class_sums, int number_of_examples)
{

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		for (int i = 0; i < mc_tm->number_of_classes; i++) {
			free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
			mc_tm_thread[t]->tsetlin_machines[i]->ta_state = mc_tm->tsetlin_machines[i]->ta_state;
			free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
			mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = mc_tm->tsetlin_machines[i]->clause_weights;
		}	
	}

	#pragma omp parallel for
	for (int l = 0; l < number_of_examples; l++) {
		int thread_id = omp_get_thread_num();

		unsigned int pos = l*step_size;
		// Identify class with largest output
		int max_class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[0], &X[pos]);
		int max_class = 0;
		for (int i = 0; i < mc_tm_thread[thread_id]->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		y[l] = max_class;
		class_sums[l] = max_class_sum;
	}

	for (int t = 0; t < max_threads; t++) {
		for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {	

			struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];

			free(tm_thread->clause_output);
			free(tm_thread->output_one_patches);
			free(tm_thread->feedback_to_la);
			free(tm_thread->feedback_to_clauses);
			free(tm_thread);
		}
	}

	free(mc_tm_thread);
	
	return;
}

/***********************************/
/*** Predict classes of inputs X ***/
/*** and return all class sums   ***/
/***********************************/

void mc_tm_predict_with_class_sums_2d(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int *class_sums, int number_of_examples)
{

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;
	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		for (int i = 0; i < mc_tm->number_of_classes; i++) {
			free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
			mc_tm_thread[t]->tsetlin_machines[i]->ta_state = mc_tm->tsetlin_machines[i]->ta_state;
			free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
			mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = mc_tm->tsetlin_machines[i]->clause_weights;
		}	
	}

	#pragma omp parallel for
	for (int l = 0; l < number_of_examples; l++) {
		int thread_id = omp_get_thread_num();

		unsigned int pos = l*step_size;
		// Identify class with largest output
		int max_class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[0], &X[pos]);
		int max_class = 0;
		for (int i = 0; i < mc_tm_thread[thread_id]->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
			// class_sums is a 2D array with dimensions [number_of_examples][number_of_classes]
			class_sums[l*mc_tm_thread[thread_id]->number_of_classes + i] = class_sum;
		}

		y[l] = max_class;
	}

	for (int t = 0; t < max_threads; t++) {
		for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {	

			struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];

			free(tm_thread->clause_output);
			free(tm_thread->output_one_patches);
			free(tm_thread->feedback_to_la);
			free(tm_thread->feedback_to_clauses);
			free(tm_thread);
		}
	}

	free(mc_tm_thread);
	
	return;
}


/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void mc_tm_update(struct MultiClassTsetlinMachine *mc_tm, unsigned int *Xi, int target_class)
{
	tm_update(mc_tm->tsetlin_machines[target_class], Xi, 1);

	// Randomly pick one of the other classes, for pairwise learning of class output 
	unsigned int negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	while (negative_target_class == target_class) {
		negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	}
	tm_update(mc_tm->tsetlin_machines[negative_target_class], Xi, 0);
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];

	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		mc_tm->tsetlin_machines[i]->clause_lock = (omp_lock_t *)malloc(sizeof(omp_lock_t) * tm->number_of_clauses);
		for (int j = 0; j < tm->number_of_clauses; ++j) {
			omp_init_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
		}
	}

	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		for (int i = 0; i < mc_tm->number_of_classes; i++) {
			free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
			mc_tm_thread[t]->tsetlin_machines[i]->ta_state = mc_tm->tsetlin_machines[i]->ta_state;
			free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
			mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = mc_tm->tsetlin_machines[i]->clause_weights;

			mc_tm_thread[t]->tsetlin_machines[i]->clause_lock = mc_tm->tsetlin_machines[i]->clause_lock;
		}	
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		#pragma omp parallel for
		for (int l = 0; l < number_of_examples; l++) {
			int thread_id = omp_get_thread_num();
			unsigned int pos = l*step_size;

			mc_tm_update(mc_tm_thread[thread_id], &X[pos], y[l]);
		}
	}

	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		for (int j = 0; j < tm->number_of_clauses; ++j) {
			omp_destroy_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
		}
	}

	for (int t = 0; t < max_threads; t++) {
		for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {	

			struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];

			free(tm_thread->clause_output);
			free(tm_thread->output_one_patches);
			free(tm_thread->feedback_to_la);
			free(tm_thread->feedback_to_clauses);
			free(tm_thread);
		}
	}

	free(tm->clause_lock);
	free(mc_tm_thread);
}

int mc_tm_ta_state(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta)
{
	return tm_ta_state(mc_tm->tsetlin_machines[class], clause, ta);
}

int mc_tm_ta_action(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta)
{
	return tm_ta_action(mc_tm->tsetlin_machines[class], clause, ta);
}

void mc_tm_clause_configuration(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, unsigned int *clause_configuration)
{
	for (int k = 0; k < mc_tm->tsetlin_machines[class]->number_of_features; ++k) {
		clause_configuration[k] = tm_ta_action(mc_tm->tsetlin_machines[class], clause, k);
	}

	return;
}

int mc_tm_clause_weight(struct MultiClassTsetlinMachine *mc_tm, int class, int clause)
{
	return(mc_tm->tsetlin_machines[class]->clause_weights[clause]);
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void mc_tm_get_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_get_ta_state(mc_tm->tsetlin_machines[class], ta_state);
	tm_get_clause_weights(mc_tm->tsetlin_machines[class], clause_weights);

	return;
}

void mc_tm_set_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_set_ta_state(mc_tm->tsetlin_machines[class], ta_state);
	tm_set_clause_weights(mc_tm->tsetlin_machines[class], clause_weights);

	return;
}

/*******************************************************************************
**** Clause Based Transformation of Input Examples for Multi-layer Learning ****
The transformation essentially creates a new binary feature for each clause in 
each class. The value of each new feature is determined by whether the 
corresponding clause evaluates to true or false for the input example, with the
possibility of inverting this relationship based on the invert parameter. Thus,
the output shape when unraveled is 
(number_of_examples, number_of_classes * number_of_clauses).
*******************************************************************************/

void mc_tm_transform(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples)
{
	// Calculate how much memory is needed for each step
	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;
	
	// Recreate the Tsetlin Machine for each thread using the same state and weights as the original machine
	// This block is mostly memory allocation and copying of the state and weights
	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits, tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
		for (int i = 0; i < mc_tm->number_of_classes; i++) {
			free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state); // Free the ta_state
			mc_tm_thread[t]->tsetlin_machines[i]->ta_state = mc_tm->tsetlin_machines[i]->ta_state; // Copy the ta_state
			free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights); // Free the clause_weights
			mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = mc_tm->tsetlin_machines[i]->clause_weights; // Copy the clause_weights
		}	
	}
	
	#pragma omp parallel for // Parallelize the loop over examples
	for (int l = 0; l < number_of_examples; l++) {
		// Get the thread id
		int thread_id = omp_get_thread_num();
		// Get the position of the example in the input array
		unsigned int pos = l*step_size;
		
		// Process each class
		for (int i = 0; i < mc_tm->number_of_classes; i++) {	
			// Calculate clause score for each class. I don't know where the clause output is stored.
			tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos]);

			for (int j = 0; j < mc_tm->tsetlin_machines[i]->number_of_clauses; ++j) {
				// Calculate position in transformed feature space
				unsigned long transformed_feature = l*mc_tm->number_of_classes*mc_tm->tsetlin_machines[i]->number_of_clauses + i*mc_tm->tsetlin_machines[i]->number_of_clauses + j;
					
				// Get the clause output from the bit packed format
				int clause_chunk = j / 32;
				int clause_pos = j % 32;
				int clause_output = (mc_tm_thread[thread_id]->tsetlin_machines[i]->clause_output[clause_chunk] & (1 << clause_pos)) > 0;
	
				// Store the transformed feature in the output array
				if (clause_output && !invert) {
					X_transformed[transformed_feature] = 1;
				} else if (!clause_output && invert) {
					X_transformed[transformed_feature] = 1;
				} else {
					X_transformed[transformed_feature] = 0;
				}
			} 
		}
	}

	// TODO: Free the memory allocated for the threads
	
	return;
}




