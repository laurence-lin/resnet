# resnet
first resnet model imitating from online

Final performance:
Training accuracy: 91%
Testing accuracy: 88%

Reuse variables during testing:
Use tf.get_variable() under tf.variable_scope(), then we could reuse the variables when testing without creating double variables. 
tf.Variable will automatically create new variables while name scope conflict, cost more memory when model is being reused.



Regularization:
1. Add regularizer when variable is created, 此處加上regularizer項有幾種用法:
   1.在tf.get_variable(regularizer) 加上 regularizer term
   2. If use tf.Variable, we could create regularizer, then use tf.contrib.layers.apply_regularization(regularizer, weight) to create penalty for the loss
   3. Build regularization term by hand, use tf.trainable_variables() and compute by math to get penalty
 
2. Batch Normalization:
   This could increase the converge speed, and the final result does converge fast. But this havd only slight regularization effect. 

Result: If I remove the regularization term, the training does converge slower, but final accuracy don't vary too much. It seems regularization does have its benefit, but not necessary to get the best performance. 


Moving Average:

 Calculate exponential moving average could smooth the weight, remove the noise and get a average curve representing the trend of the weight updating value. 
 But I haven't apply the ema.average() and I don't now its benefit to improve performance right now. 



