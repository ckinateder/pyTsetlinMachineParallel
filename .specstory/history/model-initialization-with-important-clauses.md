## SpecStory

## Model Initialization with Important Clauses (2/22/2025, 3:22:01 PM)

_**User**_

Create a function to initialize a model with the most important clauses of another model. Keep in mind that there could be less clauses in the model being initialized than the one being transferred.

---

_**Assistant**_

I'll help you add a function to initialize a model with the most important clauses from another model. We'll need to identify the most important clauses based on their weights and transfer them to the new model.

Here's the implementation:

```python:pyTsetlinMachineParallel/tm.py
# ... existing code ...

class MultiClassTsetlinMachine():
    # ... existing code ...
    
    def initialize_from_model(self, source_model, keep_percentage=0.5):
        """
        Initialize this model with the most important clauses from another model.
        
        Parameters:
        -----------
        source_model : MultiClassTsetlinMachine
            The source model to transfer clauses from
        keep_percentage : float
            Percentage of clauses to keep from each class (between 0 and 1)
            
        Returns:
        --------
        None
        """
        if not isinstance(source_model, MultiClassTsetlinMachine):
            raise TypeError("Source model must be a MultiClassTsetlinMachine")
            
        if self.number_of_features != source_model.number_of_features:
            raise ValueError("Source model must have same number of features")
            
        if self.number_of_clauses > source_model.number_of_clauses:
            raise ValueError("Target model cannot have more clauses than source model")
            
        # Get state of source model
        source_state = source_model.get_state()
        
        # For each class
        for i in range(self.number_of_classes):
            source_weights = source_state[i][0]  # Get clause weights for class i
            source_ta_states = source_state[i][1]  # Get TA states for class i
            
            # Calculate number of clauses to keep
            n_keep = min(self.number_of_clauses, int(source_model.number_of_clauses * keep_percentage))
            
            # Get indices of most important clauses based on weights
            important_indices = (-source_weights).argsort()[:n_keep]
            
            # Initialize target weights and states
            target_weights = np.ones(self.number_of_clauses, dtype=np.uint32)
            target_ta_states = np.zeros(self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, 
                                      dtype=np.uint32)
            
            # Copy weights and states for important clauses
            for new_idx, old_idx in enumerate(important_indices):
                # Copy weight
                target_weights[new_idx] = source_weights[old_idx]
                
                # Copy TA states for this clause
                old_start = old_idx * self.number_of_ta_chunks * self.number_of_state_bits
                old_end = (old_idx + 1) * self.number_of_ta_chunks * self.number_of_state_bits
                
                new_start = new_idx * self.number_of_ta_chunks * self.number_of_state_bits
                new_end = (new_idx + 1) * self.number_of_ta_chunks * self.number_of_state_bits
                
                target_ta_states[new_start:new_end] = source_ta_states[old_start:old_end]
            
            # Set the states for this class
            _lib.mc_tm_set_state(self.mc_tm, i, target_weights, target_ta_states)

    # ... rest of existing code ...
```

This implementation:

1. Takes a source model and a keep_percentage parameter that determines what fraction of clauses to keep

2. Performs validation checks:
   - Ensures source model is correct type
   - Checks feature dimensions match
   - Verifies target model doesn't have more clauses than source

3. For each class:
   - Gets the clause weights and TA states from source model
   - Identifies the most important clauses based on weights
   - Creates new weight and state arrays for target model
   - Copies the weights and TA states for the most important clauses
   - Sets the states in the target model

Usage example:

```python
# Create source model with 100 clauses
source_tm = MultiClassTsetlinMachine(number_of_clauses=100, ...)
source_tm.fit(X_train, y_train)

# Create target model with 50 clauses
target_tm = MultiClassTsetlinMachine(number_of_clauses=50, ...)

# Initialize target with most important clauses from source
target_tm.initialize_from_model(source_tm, keep_percentage=0.5)
```

This allows you to transfer knowledge from a larger model to a smaller one by keeping only the most important clauses based on their weights. The keep_percentage parameter lets you control how many clauses to transfer.

The transferred clauses maintain their learned patterns and weights, while any remaining clauses in the target model are initialized to their default states. This can be useful for model compression or for initializing new models with pre-learned patterns.

---

_**User**_

Fix this error

---

_**Assistant**_

Ah, I see the issue. The `number_of_features` attribute isn't set until after the first `fit()` call. We need to modify the `initialize_from_model` function to handle this case and properly initialize the features.

Here's the fix:

```python:pyTsetlinMachineParallel/tm.py
def initialize_from_model(self, source_model, keep_percentage=0.5):
    """
    Initialize this model with the most important clauses from another model.
    
    Parameters:
    -----------
    source_model : MultiClassTsetlinMachine
        The source model to transfer clauses from
    keep_percentage : float
        Percentage of clauses to keep from each class (between 0 and 1)
        
    Returns:
    --------
    None
    """
    if not isinstance(source_model, MultiClassTsetlinMachine):
        raise TypeError("Source model must be a MultiClassTsetlinMachine")
        
    if self.number_of_clauses > source_model.number_of_clauses:
        raise ValueError("Target model cannot have more clauses than source model")
    
    # Initialize features and structure if not already done
    if self.mc_tm is None:
        self.number_of_classes = source_model.number_of_classes
        if self.append_negated:
            self.number_of_features = source_model.number_of_features
        else:
            self.number_of_features = source_model.number_of_features // 2
            
        self.number_of_patches = 1
        self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
        self.mc_tm = _lib.CreateMultiClassTsetlinMachine(
            self.number_of_classes, 
            self.number_of_clauses,
            self.number_of_features,
            1,  # number_of_patches
            self.number_of_ta_chunks,
            self.number_of_state_bits,
            self.T,
            self.s,
            self.s_range,
            self.boost_true_positive_feedback,
            self.weighted_clauses
        )
        
    # Get state of source model
    source_state = source_model.get_state()
    
    # For each class
    for i in range(self.number_of_classes):
        source_weights = source_state[i][0]  # Get clause weights for class i
        source_ta_states = source_state[i][1]  # Get TA states for class i
        
        # Calculate number of clauses to keep
        n_keep = min(self.number_of_clauses, int(source_model.number_of_clauses * keep_percentage))
        
        # Get indices of most important clauses based on weights
        important_indices = (-source_weights).argsort()[:n_keep]
        
        # Initialize target weights and states
        target_weights = np.ones(self.number_of_clauses, dtype=np.uint32)
        target_ta_states = np.zeros(self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, 
                                  dtype=np.uint32)
        
        # Copy weights and states for important clauses
        for new_idx, old_idx in enumerate(important_indices):
            # Copy weight
            target_weights[new_idx] = source_weights[old_idx]
            
            # Copy TA states for this clause
            old_start = old_idx * self.number_of_ta_chunks * self.number_of_state_bits
            old_end = (old_idx + 1) * self.number_of_ta_chunks * self.number_of_state_bits
            
            new_start = new_idx * self.number_of_ta_chunks * self.number_of_state_bits
            new_end = (new_idx + 1) * self.number_of_ta_chunks * self.number_of_state_bits
            
            target_ta_states[new_start:new_end] = source_ta_states[old_start:old_end]
        
        # Set the states for this class
        _lib.mc_tm_set_state(self.mc_tm, i, target_weights, target_ta_states)
```

The key changes are:

1. Removed the feature number check since it wasn't initialized yet
2. Added initialization code similar to what happens in `fit()` when the model is first created
3. Properly handle the feature count based on whether negated features are appended
4. Initialize the TsetlinMachine structure if it hasn't been done yet

This should now work even if `fit()` hasn't been called on the target model yet. The function will initialize the model structure based on the source model's configuration.

---

_**User**_

Add functions for knowledge distillation. Take soft labels from he teacher as an additional input to help train the student

---

_**Assistant**_

I'll add functions to support knowledge distillation by allowing training with soft labels from a teacher model. Here's the implementation:

```python:pyTsetlinMachineParallel/tm.py
class MultiClassTsetlinMachine():
    # ... existing code ...
    
    def fit_with_distillation(self, X, y, teacher_model, epochs=100, alpha=0.5, temperature=2.0, incremental=False):
        """
        Train the model using both hard labels and soft labels from a teacher model.
        
        Parameters:
        -----------
        X : array-like
            Training examples
        y : array-like
            True labels
        teacher_model : MultiClassTsetlinMachine
            Teacher model to get soft labels from
        epochs : int
            Number of training epochs
        alpha : float
            Weight for soft labels (1-alpha is weight for hard labels)
        temperature : float
            Temperature for softening teacher predictions
        incremental : bool
            Whether to train incrementally or reinitialize
            
        Returns:
        --------
        None
        """
        number_of_examples = X.shape[0]

        # Get soft labels from teacher
        soft_labels = teacher_model.get_soft_labels(X, temperature)

        if self.mc_tm == None:
            self.number_of_classes = int(np.max(y) + 1)
            
            if self.append_negated:
                self.number_of_features = X.shape[1]*2
            else:
                self.number_of_features = X.shape[1]

            self.number_of_patches = 1
            self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
            self.mc_tm = _lib.CreateMultiClassTsetlinMachine(
                self.number_of_classes, 
                self.number_of_clauses,
                self.number_of_features,
                1,
                self.number_of_ta_chunks,
                self.number_of_state_bits,
                self.T,
                self.s,
                self.s_range,
                self.boost_true_positive_feedback,
                self.weighted_clauses
            )
        elif incremental == False:
            _lib.mc_tm_destroy(self.mc_tm)
            self.mc_tm = _lib.CreateMultiClassTsetlinMachine(
                self.number_of_classes,
                self.number_of_clauses,
                self.number_of_features,
                1,
                self.number_of_ta_chunks,
                self.number_of_state_bits,
                self.T,
                self.s,
                self.s_range,
                self.boost_true_positive_feedback,
                self.weighted_clauses
            )

        self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

        Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
        Ym = np.ascontiguousarray(y).astype(np.uint32)

        if self.append_negated:
            _lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
        else:
            _lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features, 1, 1, self.number_of_features, 1, 0)

        # Training loop with distillation
        for epoch in range(epochs):
            # Randomly permute examples each epoch
            perm = np.random.permutation(number_of_examples)
            self.encoded_X = self.encoded_X.reshape(number_of_examples, -1)[perm].reshape(-1)
            Ym = Ym[perm]
            epoch_soft_labels = soft_labels[perm]

            for i in range(number_of_examples):
                # Get current example's encoded input
                Xi = self.encoded_X[i * self.number_of_ta_chunks:(i + 1) * self.number_of_ta_chunks]
                
                # Calculate model's current output
                _, class_sums = self.predict_class_sums_2d(Xi.reshape(1, -1))
                
                # Convert to probabilities
                student_probs = np.exp(class_sums / temperature) 
                student_probs = student_probs / np.sum(student_probs)
                
                # Combine hard and soft targets
                hard_target = np.zeros(self.number_of_classes)
                hard_target[Ym[i]] = 1
                combined_target = alpha * epoch_soft_labels[i] + (1 - alpha) * hard_target
                
                # Update towards combined target
                target_class = np.argmax(combined_target)
                _lib.mc_tm_update(self.mc_tm, Xi, target_class)

    def get_soft_labels(self, X, temperature=1.0):
        """
        Get soft labels (probabilities) for input X.
        
        Parameters:
        -----------
        X : array-like
            Input examples
        temperature : float
            Temperature for softening predictions
            
        Returns:
        --------
        soft_labels : array-like
            Probability distribution over classes for each example
        """
        # Get raw class scores
        _, class_sums = self.predict_class_sums_2d(X)
        
        # Apply temperature scaling and convert to probabilities
        scaled_sums = class_sums / temperature
        exp_sums = np.exp(scaled_sums)
        soft_labels = exp_sums / np.sum(exp_sums, axis=1, keepdims=True)
        
        return soft_labels

```

This implementation adds two key functions:

1. `get_soft_labels()`: Gets probability distributions over classes from the teacher model, with temperature scaling to control how "soft" the probabilities are.

2. `fit_with_distillation()`: Trains the student model using both hard labels (true classes) and soft labels from the teacher. Key features:
   - Uses temperature scaling to soften both teacher and student predictions
   - Combines hard and soft targets using the alpha parameter
   - Updates the model towards the combined target
   - Includes random permutation of examples each epoch
   - Maintains the original model initialization logic

Usage example:

```python
# Create and train teacher model
teacher = MultiClassTsetlinMachine(number_of_clauses=100, ...)
teacher.fit(X_train, y_train)

# Create student model with fewer clauses
student = MultiClassTsetlinMachine(number_of_clauses=50, ...)

# Initialize student with most important teacher clauses
student.initialize_from_model(teacher, keep_percentage=0.5)

# Train student with distillation
student.fit_with_distillation(
    X_train, 
    y_train,
    teacher_model=teacher,
    epochs=100,
    alpha=0.5,  # Equal weight to hard and soft labels
    temperature=2.0  # Higher temperature = softer probabilities
)
```

The distillation process helps transfer the teacher's learned patterns to the student model, potentially allowing the smaller student model to achieve better performance than if it was trained on hard labels alone.

---



---

