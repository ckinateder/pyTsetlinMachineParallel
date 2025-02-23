## SpecStory

## Implementing Knowledge Distillation in Tsetlin Machine (2/22/2025, 3:01:18 PM)

_**User**_

I want to add knowledge distillation to this tsetlin machine implementation. The distillation fit function should function like the normal fit function, but take in the soft labels from the teacher at each training step.

---

_**Assistant**_

To add knowledge distillation to your Tsetlin Machine implementation, you will need to modify the `fit` function to accept soft labels from a teacher model. This involves creating a new function, `fit_with_distillation`, which will use both the hard labels and the soft labels during training. Here's how you can implement this:

### Step 1: Modify the Python Wrapper

Add a new method `fit_with_distillation` to the `MultiClassTsetlinMachine` class in `tm.py`.

```python:pyTsetlinMachineParallel/tm.py
class MultiClassTsetlinMachine():
    # existing code...

    def fit_with_distillation(self, X, Y, soft_labels, epochs=100, alpha=0.5, incremental=False):
        """
        Fit the Tsetlin Machine using knowledge distillation.
        
        Parameters:
        - X: Input features.
        - Y: Hard labels.
        - soft_labels: Soft labels from the teacher model.
        - epochs: Number of training epochs.
        - alpha: Weighting factor for the soft labels.
        - incremental: Whether to use incremental training.
        """
        number_of_examples = X.shape[0]

        self.number_of_classes = int(np.max(Y) + 1)

        if self.mc_tm == None:
            if self.append_negated:
                self.number_of_features = X.shape[1]*2
            else:
                self.number_of_features = X.shape[1]

            self.number_of_patches = 1
            self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
            self.mc_tm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
        elif incremental == False:
            _lib.mc_tm_destroy(self.mc_tm)
            self.mc_tm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)

        self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

        Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
        Ym = np.ascontiguousarray(Y).astype(np.uint32)
        soft_labels = np.ascontiguousarray(soft_labels).astype(np.float32)

        if self.append_negated:
            _lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
        else:
            _lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features, 1, 1, self.number_of_features, 1, 0)

        _lib.mc_tm_fit_with_distillation(self.mc_tm, self.encoded_X, Ym, soft_labels, number_of_examples, epochs, alpha)

        return
```

### Step 2: Update the C Code

Add a new function `mc_tm_fit_with_distillation` in `MultiClassConvolutionalTsetlinMachine.c` to handle the distillation process.

```c:pyTsetlinMachineParallel/MultiClassConvolutionalTsetlinMachine.c
void mc_tm_fit_with_distillation(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, float *soft_labels, int number_of_examples, int epochs, float alpha)
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

            // Combine hard and soft labels
            float combined_label = alpha * y[l] + (1 - alpha) * soft_labels[l * mc_tm->number_of_classes + y[l]];

            mc_tm_update(mc_tm_thread[thread_id], &X[pos], combined_label);
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
```

### Step 3: Update the Header File

Add the function prototype to `MultiClassConvolutionalTsetlinMachine.h`.

```c:pyTsetlinMachineParallel/MultiClassConvolutionalTsetlinMachine.h
void mc_tm_fit_with_distillation(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, float *soft_labels, int number_of_examples, int epochs, float alpha);
```

### Explanation

- **Python Wrapper**: The `fit_with_distillation` method in the Python class takes an additional `soft_labels` parameter and a weighting factor `alpha` to balance between hard and soft labels.
- **C Implementation**: The `mc_tm_fit_with_distillation` function in C handles the training process using both hard and soft labels. It combines them using the `alpha` parameter to create a weighted label for each example.
- **Header File**: The new function is declared in the header file to ensure it can be used across different C files.

This setup allows you to train the Tsetlin Machine using knowledge distillation, leveraging both the hard labels and the soft labels provided by a teacher model.

---

_**User**_

This did not work. accuracy is way lower

---

_**Assistant**_

If the accuracy is significantly lower when using knowledge distillation, there could be several reasons for this. Here are some potential issues and suggestions for improvement:

1. **Alpha Parameter**: The `alpha` parameter controls the balance between the hard labels and the soft labels. If `alpha` is not set correctly, it might not be effectively leveraging the soft labels. You might want to experiment with different values of `alpha` to find a better balance.

2. **Soft Labels**: Ensure that the soft labels are correctly computed and normalized. They should represent the probabilities of each class as predicted by the teacher model. If they are not properly normalized, the distillation process might not work as expected.

3. **Temperature in Softmax**: The temperature parameter in the softmax function can significantly affect the soft labels. A higher temperature can make the soft labels more informative by smoothing the probabilities. Experiment with different temperature values to see if it improves performance.

4. **Implementation Errors**: Double-check the implementation to ensure that the soft labels are being used correctly in the training process. Ensure that the `combined_label` calculation is correct and that it is being used appropriately in the update function.

5. **Training Process**: Ensure that the training process is correctly using the combined labels. The update function should be modified to handle the combined labels properly.

6. **Data Preprocessing**: Ensure that the input data is preprocessed in the same way for both the teacher and student models. Any discrepancies in preprocessing can lead to poor performance.

7. **Model Capacity**: The student model might not have enough capacity to learn from the teacher model. Consider increasing the complexity of the student model if possible.

8. **Debugging**: Add logging or debugging statements to check the values of the soft labels, combined labels, and any other critical variables during training to ensure they are as expected.

If you have specific parts of the code or logs that you suspect might be causing issues, feel free to share them, and I can help you debug further.

---



---

