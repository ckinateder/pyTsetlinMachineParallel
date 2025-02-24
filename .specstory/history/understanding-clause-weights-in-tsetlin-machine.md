## SpecStory

## Understanding Clause Weights in Tsetlin Machine (2/23/2025, 5:08:33 PM)

_**User**_

Why are the clause_weights always 1?

---

_**Assistant**_

<think>
Okay, let's try to figure out why the clause_weights are always 1 in the provided code. 

First, I'll look at the code structure. The user mentioned that the clause_weights aren't changing, so I need to check where they are initialized and updated.

In the Tsetlin Machine code, specifically in the tm.py file, the MultiClassTsetlinMachine class has a fit method. When initializing the machine, if weighted_clauses is set to True, the clause weights should be used. But maybe the initialization isn't handling this correctly.

Looking at the __init__ method of MultiClassTsetlinMachine, the weighted_clauses parameter is stored. Then, in the fit method, when creating the C struct via CreateMultiClassTsetlinMachine, the weighted_clauses argument is passed. 

In the C code (MultiClassConvolutionalTsetlinMachine.c), when creating the TsetlinMachine instances, the weighted_clauses parameter is passed. The TsetlinMachine struct in ConvolutionalTsetlinMachine.h has a weighted_clauses flag. 

In the tm_update function (ConvolutionalTsetlinMachine.c), during Type I feedback, if weighted_clauses is true, the clause weight is incremented. Similarly, during Type II feedback, if weighted and the weight is greater than 1, it's decremented. 

Wait, but in the Python code examples/distill.py, when initializing the student and teacher, the parameters are set with (400, 10, 5) and (50, 10, 5). Looking at the MultiClassTsetlinMachine __init__ in tm.py, the parameters are number_of_clauses, T, s, etc. The student_params are (50, 10, 5), which would be clauses, T, s. The weighted_clauses argument is not passed here, so it defaults to False. 

Ah, that's the issue! The user probably didn't enable weighted_clauses when creating the Tsetlin Machine instances. So even if the C code supports it, the Python wrapper isn't setting the flag. Therefore, the clause_weights remain at 1 because the weighted_clauses parameter is False, so the code doesn't update them.

To fix this, when creating the Tsetlin Machine in Python, the weighted_clauses should be set to True. For example, in the student and teacher initialization, adding weighted_clauses=True would allow the clause weights to be updated. The user needs to modify the code where the Tsetlin Machines are instantiated to include this parameter.
</think>

The clause weights remain 1 because the `weighted_clauses` parameter isn't being set when creating the Tsetlin Machines. Here's how to fix it:

```python:examples/distill.py
# In the parameter definitions:
teacher_params = (400, 10, 5, 1, 8, True, False)  # Added weighted_clauses=True
student_params = (50, 10, 5, 1, 8, True, False)  # Added weighted_clauses=True

# Then modify the machine creation:
teacher = MultiClassTsetlinMachine(*teacher_params)
student = MultiClassTsetlinMachine(*student_params)
```

The parameters should be updated to include the `weighted_clauses` flag (7th parameter in the tuple). The current code uses default False values for this parameter, which disables weight updates.

The key parameters in order are:
1. number_of_clauses
2. T
3. s 
4. boost_true_positive_feedback
5. number_of_state_bits
6. append_negated
7. weighted_clauses (needs to be True for weight updates)

Without enabling `weighted_clauses`, the weight update logic in the C code remains inactive, keeping all clause weights at their initial value of 1.

---

_**User**_

Great, thanks. I notice that with weighted_clauses set to true, there is more randomness in the accuracies. Why is that?

---

_**Assistant**_

<think>
Okay, the user is asking why enabling weighted clauses introduces more randomness in the accuracies. Let me start by recalling what weighted clauses do. When weighted_clauses is True, each clause's vote has a weight that can be incremented or decremented during training. This adds another layer of complexity to the model. 

In the original setup without weighted clauses, all clauses contribute equally (weight of 1). But with weighting, the model now has to learn both the clause configurations and their weights. This additional parameter can make the model more flexible but also more sensitive to initial conditions and training dynamics.

I remember that in the Tsetlin Machine, clauses are updated based on feedback. With weights, every time a clause is used correctly, its weight might increase, and if it's incorrect, it might decrease. This means that the importance of each clause isn't static anymore. During the early stages of training, these weight adjustments can vary significantly, leading to different clause influences each run. This variability could explain the increased randomness in accuracies.

Also, the way negative examples are sampled during training might interact with the weights. The selection of negative classes based on soft labels could lead to different clauses being reinforced or penalized in each run, especially since the weights are being adjusted dynamically. This stochastic process in selecting negative examples combined with weight updates might amplify the randomness.

Another factor is the initialization of clause weights. Even though they start at 1, as training progresses, some clauses might get much higher weights, making the model's predictions more dependent on those clauses. If the initial clauses that get boosted vary between runs, the overall model performance could differ more noticeably.

Additionally, the learning process with weighted clauses might be more prone to getting stuck in local optima. Since the weights are being adjusted, the model's path through the optimization landscape could vary more between different training sessions, leading to varied final accuracies.

I should also consider the impact of the training epochs. With more parameters to learn (the weights), the model might require more epochs to stabilize. If the user is using the same number of epochs as before, the model might not have converged yet, leading to higher variance in results.

In summary, the introduction of clause weights adds more trainable parameters and dynamic adjustments, which can increase model variance. The interaction between weight updates, stochastic negative sampling, and initial conditions contributes to the observed randomness in accuracies when weighted_clauses is enabled.
</think>

The increased randomness with weighted clauses stems from three key factors in the Tsetlin Machine architecture:

1. **Dynamic Clause Weight Updates** 🔄
When `weighted_clauses=True`, clause weights become trainable parameters that are updated during:
```python
# From

---



---

