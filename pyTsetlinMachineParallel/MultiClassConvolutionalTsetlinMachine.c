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
#include "fast_rand.h"
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
		int max_class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[0], &X[pos], 1);
		int max_class = 0;
		for (int i = 0; i < mc_tm_thread[thread_id]->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos], 1);
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
/*** and return all class sums   ***/
/*** THERE IS NO CLAMPING HERE ***/
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
		int max_class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[0], &X[pos], 0);
		int max_class = 0;
		for (int i = 0; i < mc_tm_thread[thread_id]->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos], 0);
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
	// This block is just memory allocation and copying of the state and weights
	int max_threads = omp_get_max_threads();
	struct MultiClassTsetlinMachine **mc_tm_thread = (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
	struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
	for (int t = 0; t < max_threads; t++) {
		mc_tm_thread[t] = CreateMultiClassTsetlinMachine(mc_tm->number_of_classes, tm->number_of_clauses, tm->number_of_features, 
															tm->number_of_patches, tm->number_of_ta_chunks, tm->number_of_state_bits,
															tm->T, tm->s, tm->s_range, tm->boost_true_positive_feedback, tm->weighted_clauses);
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
			// Calculate clause score for each class. Clause outputs are stored in tm
			tm_score(mc_tm_thread[thread_id]->tsetlin_machines[i], &X[pos], 1);

			for (int j = 0; j < mc_tm->tsetlin_machines[i]->number_of_clauses; ++j) {
				// Calculate position in transformed feature space
				unsigned long transformed_feature = l*mc_tm->number_of_classes*mc_tm->tsetlin_machines[i]->number_of_clauses + 
													i*mc_tm->tsetlin_machines[i]->number_of_clauses + j;
					
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
// Add new function for soft label training
void mc_tm_fit_soft(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, float *soft_labels, int number_of_examples, int epochs)
{
    /*------------------------------------------------------------*/
    /* Initialization and Setup                                   */
    /*------------------------------------------------------------*/
    // Calculate input data stride between consecutive examples
    unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

    // Get maximum available threads for parallel processing
    int max_threads = omp_get_max_threads();
    
    // Allocate thread-local TM instances
    struct MultiClassTsetlinMachine **mc_tm_thread = 
        (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
    struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
    
    /*------------------------------------------------------------*/
    /* Clause Lock Initialization                                 */
    /*------------------------------------------------------------*/
    // Initialize OpenMP locks for thread-safe clause updates
    for (int i = 0; i < mc_tm->number_of_classes; i++) {
        // Allocate array of locks (one per clause)
        mc_tm->tsetlin_machines[i]->clause_lock = 
            (omp_lock_t *)malloc(sizeof(omp_lock_t) * tm->number_of_clauses);
        
        // Initialize each lock
        for (int j = 0; j < tm->number_of_clauses; ++j) {
            omp_init_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
        }
    }

    /*------------------------------------------------------------*/
    /* Thread-Local TM Setup                                      */
    /*------------------------------------------------------------*/
    // Create thread-specific TM instances with shared state
    for (int t = 0; t < max_threads; t++) {
        // Create skeleton TM structure
        mc_tm_thread[t] = CreateMultiClassTsetlinMachine(
            mc_tm->number_of_classes, 
            tm->number_of_clauses,
            tm->number_of_features,
            tm->number_of_patches,
            tm->number_of_ta_chunks,
            tm->number_of_state_bits,
            tm->T,
            tm->s,
            tm->s_range,
            tm->boost_true_positive_feedback,
            tm->weighted_clauses
        );
        
        // Share actual state between threads
        for (int i = 0; i < mc_tm->number_of_classes; i++) {
            // Share TA states
            free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
            mc_tm_thread[t]->tsetlin_machines[i]->ta_state = 
                mc_tm->tsetlin_machines[i]->ta_state;
            
            // Share clause weights
            free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
            mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = 
                mc_tm->tsetlin_machines[i]->clause_weights;
            
            // Share clause locks
            mc_tm_thread[t]->tsetlin_machines[i]->clause_lock = 
                mc_tm->tsetlin_machines[i]->clause_lock;
        }    
    }

    /*------------------------------------------------------------*/
    /* Main Training Loop                                          */
    /*------------------------------------------------------------*/
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Process examples in parallel using OpenMP
        #pragma omp parallel for
        for (int l = 0; l < number_of_examples; l++) {
            // Get thread ID and example position
            int thread_id = omp_get_thread_num();
            unsigned int pos = l * step_size;
            
            // Get true class label for this example
            int target_class = y[l];
            
            // Pointer to soft label probabilities (num_classes elements)
            float *example_soft_labels = &soft_labels[l * mc_tm->number_of_classes];
            
            /*----------------------------------------------------*/
            /* Positive Class Update                             */
            /*----------------------------------------------------*/
            // Always reinforce the true class with Type I feedback
            tm_update(mc_tm_thread[thread_id]->tsetlin_machines[target_class], 
                     &X[pos], 1);

            /*----------------------------------------------------*/
            /* Negative Class Selection - Improved                */
            /*----------------------------------------------------*/
            // Hybrid approach: balance between soft probabilities and randomness
            float randomization_factor = 0.3f; // 30% random selection, 70% soft probabilities
            float random_val = (float)fast_rand() / FAST_RAND_MAX;
            
            int negative_class = -1; // Initialize with invalid value
            
            if (random_val < randomization_factor) {
                // Use uniform random selection (like original algorithm)
                do {
                    negative_class = fast_rand() % mc_tm->number_of_classes;
                } while (negative_class == target_class);
            } else {
                // Calculate total probability mass for negative classes
                float total = 0.0f;
                for (int i = 0; i < mc_tm->number_of_classes; i++) {
                    if (i != target_class) {
                        total += example_soft_labels[i];
                    }
                }
                
                if (total <= 0.0001f) { // Effectively zero
                    // Fallback to random when probabilities are too small
                    do {
                        negative_class = fast_rand() % mc_tm->number_of_classes;
                    } while (negative_class == target_class);
                } else {
                    // Proportional sampling using inverse transform method
                    float threshold = (float)fast_rand() / FAST_RAND_MAX * total;
                    float cumulative = 0.0f;
                    
                    // Find first class where cumulative probability >= threshold
                    for (int i = 0; i < mc_tm->number_of_classes; i++) {
                        if (i == target_class) continue;
                        
                        cumulative += example_soft_labels[i];
                        if (cumulative >= threshold) {
                            negative_class = i;
                            break;
                        }
                    }
                    
                    // Safety check - if we somehow didn't select a class
                    if (negative_class == -1) {
                        do {
                            negative_class = fast_rand() % mc_tm->number_of_classes;
                        } while (negative_class == target_class);
                    }
                }
            }

            /*----------------------------------------------------*/
            /* Negative Class Update                             */
            /*----------------------------------------------------*/
            // Apply Type II feedback to selected negative class
            tm_update(mc_tm_thread[thread_id]->tsetlin_machines[negative_class], 
                     &X[pos], 0);
        }
    }

    /*------------------------------------------------------------*/
    /* Cleanup and Resource Release                               */
    /*------------------------------------------------------------*/
    // Destroy clause locks
    for (int i = 0; i < mc_tm->number_of_classes; i++) {
        for (int j = 0; j < tm->number_of_clauses; ++j) {
            omp_destroy_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
        }
    }

    // Free thread-local TM instances
    for (int t = 0; t < max_threads; t++) {
        for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {    
            struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];
            // Free temporary buffers while preserving shared state
            free(tm_thread->clause_output);
            free(tm_thread->output_one_patches);
            free(tm_thread->feedback_to_la);
            free(tm_thread->feedback_to_clauses);
            free(tm_thread);
        }
    }

    // Final memory cleanup
    free(tm->clause_lock);
    free(mc_tm_thread);
}

// Add new function for soft label training
/*
 * This function implements knowledge distillation for Tsetlin Machines using soft labels.
 * 
 * Parameters:
 * - mc_tm: Pointer to the multi-class Tsetlin Machine to be trained
 * - X: Input data
 * - y: True class labels
 * - soft_labels: Probability distribution from teacher model (already temperature-scaled)
 * - number_of_examples: Number of training examples
 * - epochs: Number of training epochs
 * - alpha: Balance between hard and soft labels (0.0 = all soft, 1.0 = all hard)
 * - temperature: Controls additional temperature scaling for training dynamics
 *   - Higher values make the system more sensitive to teacher confidences
 *   - A temperature around 2-4 works well for most cases
 *   - This parameter is used to sharpen/soften the teacher's probability distribution
 */
void mc_tm_fit_soft_improved(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, float *soft_labels, int number_of_examples, int epochs, float alpha, float temperature)
{
    /*------------------------------------------------------------*/
    /* Initialization and Setup                                   */
    /*------------------------------------------------------------*/
    // Calculate input data stride between consecutive examples
    unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

    // Get maximum available threads for parallel processing
    int max_threads = omp_get_max_threads();
    
    // Allocate thread-local TM instances
    struct MultiClassTsetlinMachine **mc_tm_thread = 
        (void *)malloc(sizeof(struct MultiClassTsetlinMachine *) * max_threads);
    struct TsetlinMachine *tm = mc_tm->tsetlin_machines[0];
    
    /*------------------------------------------------------------*/
    /* Clause Lock Initialization                                 */
    /*------------------------------------------------------------*/
    // Initialize OpenMP locks for thread-safe clause updates
    for (int i = 0; i < mc_tm->number_of_classes; i++) {
        // Allocate array of locks (one per clause)
        mc_tm->tsetlin_machines[i]->clause_lock = 
            (omp_lock_t *)malloc(sizeof(omp_lock_t) * tm->number_of_clauses);
        
        // Initialize each lock
        for (int j = 0; j < tm->number_of_clauses; ++j) {
            omp_init_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
        }
    }

    /*------------------------------------------------------------*/
    /* Thread-Local TM Setup                                      */
    /*------------------------------------------------------------*/
    // Create thread-specific TM instances with shared state
    for (int t = 0; t < max_threads; t++) {
        // Create skeleton TM structure
        mc_tm_thread[t] = CreateMultiClassTsetlinMachine(
            mc_tm->number_of_classes, 
            tm->number_of_clauses,
            tm->number_of_features,
            tm->number_of_patches,
            tm->number_of_ta_chunks,
            tm->number_of_state_bits,
            tm->T,
            tm->s,
            tm->s_range,
            tm->boost_true_positive_feedback,
            tm->weighted_clauses
        );
        
        // Share actual state between threads
        for (int i = 0; i < mc_tm->number_of_classes; i++) {
            // Share TA states
            free(mc_tm_thread[t]->tsetlin_machines[i]->ta_state);
            mc_tm_thread[t]->tsetlin_machines[i]->ta_state = 
                mc_tm->tsetlin_machines[i]->ta_state;
            
            // Share clause weights
            free(mc_tm_thread[t]->tsetlin_machines[i]->clause_weights);
            mc_tm_thread[t]->tsetlin_machines[i]->clause_weights = 
                mc_tm->tsetlin_machines[i]->clause_weights;
            
            // Share clause locks
            mc_tm_thread[t]->tsetlin_machines[i]->clause_lock = 
                mc_tm->tsetlin_machines[i]->clause_lock;
        }    
    }

    /*------------------------------------------------------------*/
    /* Main Training Loop                                         */
    /*------------------------------------------------------------*/
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Process examples in parallel using OpenMP
        #pragma omp parallel for
        for (int l = 0; l < number_of_examples; l++) {
            // Get thread ID and example position
            int thread_id = omp_get_thread_num();
            unsigned int pos = l * step_size;
            
            // Get true class label for this example
            int target_class = y[l];
            
            // Apply temperature scaling to soft labels (we already have softmax probabilities)
            // Use a squared temperature for the same effect as double scaling
            float adjusted_temperature = temperature * temperature;
            float scaled_probs[mc_tm->number_of_classes];
            float sum = 0.0;
            
            // Single temperature scaling with adjusted temperature gives same effect
            for (int i = 0; i < mc_tm->number_of_classes; i++) {
                scaled_probs[i] = powf(soft_labels[l * mc_tm->number_of_classes + i], 1.0/adjusted_temperature);
                sum += scaled_probs[i];
            }
            
            // Single normalization
            for (int i = 0; i < mc_tm->number_of_classes; i++) 
                scaled_probs[i] /= sum;
            
            /*----------------------------------------------------*/
            /* Hard Label (True Class) Training                   */
            /*----------------------------------------------------*/
            // Apply Type I feedback to true class with probability based on alpha
            //if ((float)fast_rand() / FAST_RAND_MAX <= alpha) {
                tm_update(mc_tm_thread[thread_id]->tsetlin_machines[target_class], 
                         &X[pos], 1);
            //}
            
            /*----------------------------------------------------*/
            /* Soft Label Training for All Classes                */
            /*----------------------------------------------------*/
            // Process ALL classes using soft labels
            for (int i = 0; i < mc_tm->number_of_classes; i++) {
                // Skip true class if already trained with hard labels
                if (i == target_class && alpha > 0.0) continue;
                
                // Calculate base feedback probability based on soft label
                float feedback_prob = (1.0 - alpha) * scaled_probs[i];
                
                // No training if probability is too small
                if (feedback_prob < 0.001) continue;
                
                // Determine if we should give positive or negative feedback
                // For high probabilities from teacher, train with positive feedback (1)
                // For low probabilities, train with negative feedback (0)
                int feedback_type = 0;
                if (scaled_probs[i] > 0.5) {
                    feedback_type = 1;
                    // Strengthen positive feedback based on teacher confidence
                    // Higher temperature makes this more aggressive
                    feedback_prob = feedback_prob * (1.0 + scaled_probs[i] * temperature);
                } else {
                    feedback_type = 0;
                    // Strengthen negative feedback based on teacher confidence
                    // Higher temperature makes this more aggressive
                    feedback_prob = feedback_prob * (1.0 + (1.0 - scaled_probs[i]) * temperature);
                }
                
                // Apply feedback with probability proportional to soft label strength
                if ((float)fast_rand() / FAST_RAND_MAX <= feedback_prob) {
                    tm_update(mc_tm_thread[thread_id]->tsetlin_machines[i], 
                             &X[pos], feedback_type);
                }
            }
        }
    }

    /*------------------------------------------------------------*/
    /* Cleanup and Resource Release                               */
    /*------------------------------------------------------------*/
    // Destroy clause locks
    for (int i = 0; i < mc_tm->number_of_classes; i++) {
        for (int j = 0; j < tm->number_of_clauses; ++j) {
            omp_destroy_lock(&mc_tm->tsetlin_machines[i]->clause_lock[j]);
        }
    }

    // Free thread-local TM instances
    for (int t = 0; t < max_threads; t++) {
        for (int i = 0; i < mc_tm_thread[t]->number_of_classes; i++) {    
            struct TsetlinMachine *tm_thread = mc_tm_thread[t]->tsetlin_machines[i];
            // Free temporary buffers while preserving shared state
            free(tm_thread->clause_output);
            free(tm_thread->output_one_patches);
            free(tm_thread->feedback_to_la);
            free(tm_thread->feedback_to_clauses);
            free(tm_thread);
        }
    }

    // Final memory cleanup
    free(tm->clause_lock);
    free(mc_tm_thread);
}