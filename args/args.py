# TODO : Arguments
args = {}
args['dim_h']                 = 64                    
args['n_channel']             = 3                     
args['n_z']                   = 400                   # number of dimensions in latent space. [optional: 600]
args['lr']                    = 0.0002                # learning rate for Adam optimizer .0002
args['epochs']                = 50                    
args['batch_size']            = 64                    # batch size for SGD, Adam
args['save']                  = True                  # save weights at each epoch of training if True
args['train']                 = True                  
args['resnet_learning_rate']  = 0.0008                # learning rate for Adam optimizer .0008 with resnet18
args['resnet_num_epochs']     = 50                    
args['oversampling_method']   = 'SMOTE'               # Oversampling Algorithm: ADASYN/SMOTE
args['data_name']             = 'Medical'             # Dataset
args['header']                = True
args['sigma']                 = 1.0
args['lambda']                = 0.01
args['pre_trained']           = False

# TODO : Measuring Metrics
num_repeats = 1
Accuracy      = np.zeros(num_repeats)
GeometricMean = np.zeros(num_repeats)
F1Score       = np.zeros(num_repeats)
Precision     = np.zeros(num_repeats)
Recall        = np.zeros(num_repeats)

result_df = pd.DataFrame(columns=['Accuracy', 'GeometricMean', 'F1Score', 'Precision', 'Recall'])

# TODO : Check overfitting problem
train_loss_total = []
validation_loss_total = []
train_acc_total = []
validation_acc_total = []


# TODO : Time
train_model_time           = np.zeros(num_repeats)
generate_samples_time      = np.zeros(num_repeats)
train_classification_time  = np.zeros(num_repeats)
prediction_time            = np.zeros(num_repeats)

time_df = pd.DataFrame(columns=['Train model time', 'Generate samples time', 'Train classification time', 'Prediction time'])
     