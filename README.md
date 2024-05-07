**Instructions to recreate the results** <br>
1 Extract database in the root directory <br>
2 Run test.py <br>
3 Run beam search <br>
4 If you want to train the model make sure to load the model from model.pt before running. Also have checkpoint saves if you don't know if your device will run out of memory and crash.


If you want to generate a new database, run generate_db with the desired number of test_cases and valid_cases.<br>
Do not forget to uncomment the saving statements at the very bottom.<br>

If you want to make changes to the neural network model, you can find the whole architecture in neural_network.py<br>

For training, all the parameters are set at the top like num_updates(epochs), how many stacks do you want to train the network on = lenght. After how many epochs do you wanna see the results(loss,accu) update_check.
batch_size.<br>

For beam search go to beam_search.py and run beam search for proofs in test.


