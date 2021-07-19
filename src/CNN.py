import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, add, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

def conv_block(input_tensor,
               kernel_size,
               nb_filter,
               name
               ):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding="same", activation="relu", name=name + "_conv1")(
        input_tensor)

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding="same", name=name + "_conv2")(x)

    # Compute one convolution to resize correctly the input_tensor as to compute the add operation

    input_tensor = Conv2D(nb_filter, (1, 1), padding="same")(input_tensor)

    x = add([x, input_tensor])

    x1 = Activation('relu')(x)

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding="same", activation="relu", name=name + "_conv3")(x1)

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding="same", name=name + "_conv4")(x)

    x = add([x, x1])

    x = Activation('relu')(x)

    return x

def model_technicolor_vector_multi_qp():
    img_input = Input(shape=(68, 68, 1))
    qp_input = Input(shape=(1,), name='qp_input')

    x = Conv2D(16, (3, 3), padding="same", activation= "relu", name='init_conv')(img_input)

    x = MaxPooling2D((2, 2), padding="same", strides=(2, 2))(x)

    x = conv_block(x, 3, 24, name='block1')

    x = MaxPooling2D((3, 3), strides=(3, 3), padding="same")(x)

    x = conv_block(x, 3, 32, name='block2')

    x = MaxPooling2D((3, 3), strides=(3, 3), padding="same")(x)

    x = conv_block(x, 3, 48, name='block3')

    x = MaxPooling2D((3, 3), strides=(3, 3), padding="same")(x)

    x = Flatten()(x)
    
    x = Concatenate()([x, qp_input])

    output_luma = Dense(480, activation='sigmoid', name='fc480')(x)

    # Add QP to X here before creating the model

    model = Model(inputs = [img_input, qp_input], outputs = output_luma, name='archiGlobal')

    return model
