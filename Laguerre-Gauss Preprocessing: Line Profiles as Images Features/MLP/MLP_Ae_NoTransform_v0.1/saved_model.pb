��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d388ч
�
input_layer/kernelVarHandleOp*
shape:
��*#
shared_nameinput_layer/kernel*
dtype0*
_output_shapes
: 
{
&input_layer/kernel/Read/ReadVariableOpReadVariableOpinput_layer/kernel*
dtype0* 
_output_shapes
:
��
y
input_layer/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*!
shared_nameinput_layer/bias
r
$input_layer/bias/Read/ReadVariableOpReadVariableOpinput_layer/bias*
dtype0*
_output_shapes	
:�
�
hidden_layer_1/kernelVarHandleOp*
shape:	�@*&
shared_namehidden_layer_1/kernel*
dtype0*
_output_shapes
: 
�
)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
dtype0*
_output_shapes
:	�@
~
hidden_layer_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*$
shared_namehidden_layer_1/bias
w
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
dtype0*
_output_shapes
:@
�
hidden_layer_2/kernelVarHandleOp*
shape
:@ *&
shared_namehidden_layer_2/kernel*
dtype0*
_output_shapes
: 

)hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel*
dtype0*
_output_shapes

:@ 
~
hidden_layer_2/biasVarHandleOp*
shape: *$
shared_namehidden_layer_2/bias*
dtype0*
_output_shapes
: 
w
'hidden_layer_2/bias/Read/ReadVariableOpReadVariableOphidden_layer_2/bias*
dtype0*
_output_shapes
: 
�
output_layer/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
: *$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
dtype0*
_output_shapes

: 
z
output_layer/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
dtype0*
_output_shapes
:
l
RMSprop/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
dtype0	*
_output_shapes
: 
n
RMSprop/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
dtype0*
_output_shapes
: 
~
RMSprop/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
dtype0*
_output_shapes
: 
t
RMSprop/momentumVarHandleOp*
shape: *!
shared_nameRMSprop/momentum*
dtype0*
_output_shapes
: 
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
dtype0*
_output_shapes
: 
j
RMSprop/rhoVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
RMSprop/input_layer/kernel/rmsVarHandleOp*
shape:
��*/
shared_name RMSprop/input_layer/kernel/rms*
dtype0*
_output_shapes
: 
�
2RMSprop/input_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/kernel/rms*
dtype0* 
_output_shapes
:
��
�
RMSprop/input_layer/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*-
shared_nameRMSprop/input_layer/bias/rms
�
0RMSprop/input_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/bias/rms*
dtype0*
_output_shapes	
:�
�
!RMSprop/hidden_layer_1/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�@*2
shared_name#!RMSprop/hidden_layer_1/kernel/rms
�
5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_1/kernel/rms*
dtype0*
_output_shapes
:	�@
�
RMSprop/hidden_layer_1/bias/rmsVarHandleOp*
shape:@*0
shared_name!RMSprop/hidden_layer_1/bias/rms*
dtype0*
_output_shapes
: 
�
3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_1/bias/rms*
dtype0*
_output_shapes
:@
�
!RMSprop/hidden_layer_2/kernel/rmsVarHandleOp*
shape
:@ *2
shared_name#!RMSprop/hidden_layer_2/kernel/rms*
dtype0*
_output_shapes
: 
�
5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_2/kernel/rms*
dtype0*
_output_shapes

:@ 
�
RMSprop/hidden_layer_2/bias/rmsVarHandleOp*
shape: *0
shared_name!RMSprop/hidden_layer_2/bias/rms*
dtype0*
_output_shapes
: 
�
3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_2/bias/rms*
dtype0*
_output_shapes
: 
�
RMSprop/output_layer/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape
: *0
shared_name!RMSprop/output_layer/kernel/rms
�
3RMSprop/output_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/kernel/rms*
dtype0*
_output_shapes

: 
�
RMSprop/output_layer/bias/rmsVarHandleOp*
shape:*.
shared_nameRMSprop/output_layer/bias/rms*
dtype0*
_output_shapes
: 
�
1RMSprop/output_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/bias/rms*
dtype0*
_output_shapes
:

NoOpNoOp
�.
ConstConst"/device:CPU:0*�-
value�-B�- B�-
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�
7iter
	8decay
9learning_rate
:momentum
;rho	rmsl	rmsm	rmsn	rmso	'rmsp	(rmsq	1rmsr	2rmss
8
0
1
2
3
'4
(5
16
27
8
0
1
2
3
'4
(5
16
27
 
�

	variables
trainable_variables
<metrics
=layer_regularization_losses

>layers
?non_trainable_variables
regularization_losses
 
 
 
 
�
	variables
trainable_variables
@metrics
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
regularization_losses
^\
VARIABLE_VALUEinput_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEinput_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
trainable_variables
Dmetrics
Elayer_regularization_losses

Flayers
Gnon_trainable_variables
regularization_losses
 
 
 
�
	variables
trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
Knon_trainable_variables
regularization_losses
a_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
 trainable_variables
Lmetrics
Mlayer_regularization_losses

Nlayers
Onon_trainable_variables
!regularization_losses
 
 
 
�
#	variables
$trainable_variables
Pmetrics
Qlayer_regularization_losses

Rlayers
Snon_trainable_variables
%regularization_losses
a_
VARIABLE_VALUEhidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
�
)	variables
*trainable_variables
Tmetrics
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
+regularization_losses
 
 
 
�
-	variables
.trainable_variables
Xmetrics
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
/regularization_losses
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
�
3	variables
4trainable_variables
\metrics
]layer_regularization_losses

^layers
_non_trainable_variables
5regularization_losses
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

`0
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	atotal
	bcount
c
_fn_kwargs
d	variables
etrainable_variables
fregularization_losses
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1
 
 
�
d	variables
etrainable_variables
hmetrics
ilayer_regularization_losses

jlayers
knon_trainable_variables
fregularization_losses
 
 
 

a0
b1
��
VARIABLE_VALUERMSprop/input_layer/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/input_layer/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!RMSprop/hidden_layer_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/hidden_layer_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!RMSprop/hidden_layer_2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/hidden_layer_2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/output_layer/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/output_layer/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
!serving_default_input_layer_inputPlaceholder*
shape:����������*
dtype0*(
_output_shapes
:����������
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/bias*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*
Tin
2	*-
_gradient_op_typePartitionedCall-208224*-
f(R&
$__inference_signature_wrapper_207854
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input_layer/kernel/Read/ReadVariableOp$input_layer/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp)hidden_layer_2/kernel/Read/ReadVariableOp'hidden_layer_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2RMSprop/input_layer/kernel/rms/Read/ReadVariableOp0RMSprop/input_layer/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOp3RMSprop/output_layer/kernel/rms/Read/ReadVariableOp1RMSprop/output_layer/bias/rms/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-208269*(
f#R!
__inference__traced_save_208268*
Tout
2*-
config_proto

GPU

CPU2*0J 8*$
Tin
2	*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/input_layer/kernel/rmsRMSprop/input_layer/bias/rms!RMSprop/hidden_layer_1/kernel/rmsRMSprop/hidden_layer_1/bias/rms!RMSprop/hidden_layer_2/kernel/rmsRMSprop/hidden_layer_2/bias/rmsRMSprop/output_layer/kernel/rmsRMSprop/output_layer/bias/rms*-
_gradient_op_typePartitionedCall-208351*+
f&R$
"__inference__traced_restore_208350*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *#
Tin
2��
�
a
(__inference_dropout_layer_call_fn_208045

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
G__inference_input_layer_layer_call_and_return_conditional_losses_207507

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
b
C__inference_dropout_layer_call_and_return_conditional_losses_207544

inputs
identity�Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:����������j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_208040

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
F
*__inference_dropout_1_layer_call_fn_208103

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207635*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������@*
Tin
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*&
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�	
�
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208114

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�%
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207741
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity��dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�&hidden_layer_1/StatefulPartitionedCall�&hidden_layer_2/StatefulPartitionedCall�#input_layer/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507�
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544�
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616�
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207699*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688*
Tout
2�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
�\
�
"__inference__traced_restore_208350
file_prefix'
#assignvariableop_input_layer_kernel'
#assignvariableop_1_input_layer_bias,
(assignvariableop_2_hidden_layer_1_kernel*
&assignvariableop_3_hidden_layer_1_bias,
(assignvariableop_4_hidden_layer_2_kernel*
&assignvariableop_5_hidden_layer_2_bias*
&assignvariableop_6_output_layer_kernel(
$assignvariableop_7_output_layer_bias#
assignvariableop_8_rmsprop_iter$
 assignvariableop_9_rmsprop_decay-
)assignvariableop_10_rmsprop_learning_rate(
$assignvariableop_11_rmsprop_momentum#
assignvariableop_12_rmsprop_rho
assignvariableop_13_total
assignvariableop_14_count6
2assignvariableop_15_rmsprop_input_layer_kernel_rms4
0assignvariableop_16_rmsprop_input_layer_bias_rms9
5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rms7
3assignvariableop_18_rmsprop_hidden_layer_1_bias_rms9
5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rms7
3assignvariableop_20_rmsprop_hidden_layer_2_bias_rms7
3assignvariableop_21_rmsprop_output_layer_kernel_rms5
1assignvariableop_22_rmsprop_output_layer_bias_rms
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*%
dtypes
2	*p
_output_shapes^
\:::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_input_layer_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0*
dtype0	*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:{
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp2assignvariableop_15_rmsprop_input_layer_kernel_rmsIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_rmsprop_input_layer_bias_rmsIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rmsIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_rmsprop_hidden_layer_1_bias_rmsIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rmsIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp3assignvariableop_20_rmsprop_hidden_layer_2_bias_rmsIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_rmsprop_output_layer_kernel_rmsIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp1assignvariableop_22_rmsprop_output_layer_bias_rmsIdentity_22:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : 
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_208146

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :& "
 
_user_specified_nameinputs
�
c
*__inference_dropout_2_layer_call_fn_208151

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207699*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:���������@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:���������@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*&
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
F
*__inference_dropout_2_layer_call_fn_208156

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� `
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*&
_input_shapes
:��������� :& "
 
_user_specified_nameinputs
�	
�
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
-__inference_output_layer_layer_call_fn_208174

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*
Tin
2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�+
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207971

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��%hidden_layer_1/BiasAdd/ReadVariableOp�$hidden_layer_1/MatMul/ReadVariableOp�%hidden_layer_2/BiasAdd/ReadVariableOp�$hidden_layer_2/MatMul/ReadVariableOp�"input_layer/BiasAdd/ReadVariableOp�!input_layer/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:����������o
dropout/IdentityIdentityinput_layer/Relu:activations:0*
T0*(
_output_shapes
:�����������
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�@�
hidden_layer_1/MatMulMatMuldropout/Identity:output:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@�
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@s
dropout_1/IdentityIdentity!hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ �
hidden_layer_2/MatMulMatMuldropout_1/Identity:output:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: �
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� s
dropout_2/IdentityIdentity!hidden_layer_2/Relu:activations:0*
T0*'
_output_shapes
:��������� �
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: �
output_layer/MatMulMatMuldropout_2/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : : : : : 
�%
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207784

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity��dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�&hidden_layer_1/StatefulPartitionedCall�&hidden_layer_2/StatefulPartitionedCall�#input_layer/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507�
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2�
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207699*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688*
Tout
2�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : 
�!
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207819

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity��&hidden_layer_1/StatefulPartitionedCall�&hidden_layer_2/StatefulPartitionedCall�#input_layer/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:�����������
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2�
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207635*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������@*
Tin
2�
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2�
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:��������� *
Tin
2*-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*
Tin
2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : 
�[
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207936

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��%hidden_layer_1/BiasAdd/ReadVariableOp�$hidden_layer_1/MatMul/ReadVariableOp�%hidden_layer_2/BiasAdd/ReadVariableOp�$hidden_layer_2/MatMul/ReadVariableOp�"input_layer/BiasAdd/ReadVariableOp�!input_layer/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:����������Y
dropout/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: c
dropout/dropout/ShapeShapeinput_layer/Relu:activations:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������Z
dropout/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*(
_output_shapes
:�����������
dropout/dropout/mulMulinput_layer/Relu:activations:0dropout/dropout/truediv:z:0*
T0*(
_output_shapes
:�����������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:�����������
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�@�
hidden_layer_1/MatMulMatMuldropout/dropout/mul_1:z:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@�
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@[
dropout_1/dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: h
dropout_1/dropout/ShapeShape!hidden_layer_1/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:���������@�
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:���������@�
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:���������@\
dropout_1/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:���������@�
dropout_1/dropout/mulMul!hidden_layer_1/Relu:activations:0dropout_1/dropout/truediv:z:0*
T0*'
_output_shapes
:���������@�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:���������@�
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ �
hidden_layer_2/MatMulMatMuldropout_1/dropout/mul_1:z:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: �
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� [
dropout_2/dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: h
dropout_2/dropout/ShapeShape!hidden_layer_2/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:��������� �
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:��������� �
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:��������� \
dropout_2/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_2/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*'
_output_shapes
:��������� �
dropout_2/dropout/mulMul!hidden_layer_2/Relu:activations:0dropout_2/dropout/truediv:z:0*
T0*'
_output_shapes
:��������� �
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:��������� �
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� �
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: �
output_layer/MatMulMatMuldropout_2/dropout/mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
�4
�

__inference__traced_save_208268
file_prefix1
-savev2_input_layer_kernel_read_readvariableop/
+savev2_input_layer_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_rmsprop_input_layer_kernel_rms_read_readvariableop;
7savev2_rmsprop_input_layer_bias_rms_read_readvariableop@
<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop>
:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop@
<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop>
:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop>
:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop<
8savev2_rmsprop_output_layer_bias_rms_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_fd2e5e4d2ad2498cb9b219ac8eac9abd/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_rmsprop_input_layer_kernel_rms_read_readvariableop7savev2_rmsprop_input_layer_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop8savev2_rmsprop_output_layer_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:	�@:@:@ : : :: : : : : : : :
��:�:	�@:@:@ : : :: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: :	 :
 : : : : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : 
�

�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207997

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207820*\
fWRU
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207819�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
�
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_208088

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:���������@�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:���������@�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:���������@R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:���������@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:���������@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*&
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
,__inference_input_layer_layer_call_fn_208015

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_208167

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�	
�
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�!
�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207762
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity��&hidden_layer_1/StatefulPartitionedCall�&hidden_layer_2/StatefulPartitionedCall�#input_layer/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2�
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:�����������
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207635*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� *-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651�
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:��������� �
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: : : 
�

�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207831
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207820*\
fWRU
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207819�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: : : 
�
�
/__inference_hidden_layer_2_layer_call_fn_208121

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:��������� *
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�

�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207796
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207785*\
fWRU
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207784�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
�
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_208141

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:��������� �
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:��������� �
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:��������� R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:��������� a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:��������� i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*&
_input_shapes
:��������� :& "
 
_user_specified_nameinputs
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_207723

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :& "
 
_user_specified_nameinputs
�

�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207984

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:���������*-
_gradient_op_typePartitionedCall-207785*\
fWRU
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207784*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : 
�=
�	
!__inference__wrapped_model_207490
input_layer_inputF
Bmlp_ae_notransform_v0_1_input_layer_matmul_readvariableop_resourceG
Cmlp_ae_notransform_v0_1_input_layer_biasadd_readvariableop_resourceI
Emlp_ae_notransform_v0_1_hidden_layer_1_matmul_readvariableop_resourceJ
Fmlp_ae_notransform_v0_1_hidden_layer_1_biasadd_readvariableop_resourceI
Emlp_ae_notransform_v0_1_hidden_layer_2_matmul_readvariableop_resourceJ
Fmlp_ae_notransform_v0_1_hidden_layer_2_biasadd_readvariableop_resourceG
Cmlp_ae_notransform_v0_1_output_layer_matmul_readvariableop_resourceH
Dmlp_ae_notransform_v0_1_output_layer_biasadd_readvariableop_resource
identity��=MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp�<MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOp�=MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp�<MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOp�:MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOp�9MLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOp�;MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOp�:MLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOp�
9MLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOpReadVariableOpBmlp_ae_notransform_v0_1_input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
*MLP_Ae_NoTransform_v0.1/input_layer/MatMulMatMulinput_layer_inputAMLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOpReadVariableOpCmlp_ae_notransform_v0_1_input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
+MLP_Ae_NoTransform_v0.1/input_layer/BiasAddBiasAdd4MLP_Ae_NoTransform_v0.1/input_layer/MatMul:product:0BMLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(MLP_Ae_NoTransform_v0.1/input_layer/ReluRelu4MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
(MLP_Ae_NoTransform_v0.1/dropout/IdentityIdentity6MLP_Ae_NoTransform_v0.1/input_layer/Relu:activations:0*
T0*(
_output_shapes
:�����������
<MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOpReadVariableOpEmlp_ae_notransform_v0_1_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�@�
-MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMulMatMul1MLP_Ae_NoTransform_v0.1/dropout/Identity:output:0DMLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOpFmlp_ae_notransform_v0_1_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@�
.MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAddBiasAdd7MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul:product:0EMLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+MLP_Ae_NoTransform_v0.1/hidden_layer_1/ReluRelu7MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*MLP_Ae_NoTransform_v0.1/dropout_1/IdentityIdentity9MLP_Ae_NoTransform_v0.1/hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
<MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOpReadVariableOpEmlp_ae_notransform_v0_1_hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ �
-MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMulMatMul3MLP_Ae_NoTransform_v0.1/dropout_1/Identity:output:0DMLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
=MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOpFmlp_ae_notransform_v0_1_hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: �
.MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAddBiasAdd7MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul:product:0EMLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+MLP_Ae_NoTransform_v0.1/hidden_layer_2/ReluRelu7MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*MLP_Ae_NoTransform_v0.1/dropout_2/IdentityIdentity9MLP_Ae_NoTransform_v0.1/hidden_layer_2/Relu:activations:0*
T0*'
_output_shapes
:��������� �
:MLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOpReadVariableOpCmlp_ae_notransform_v0_1_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: �
+MLP_Ae_NoTransform_v0.1/output_layer/MatMulMatMul3MLP_Ae_NoTransform_v0.1/dropout_2/Identity:output:0BMLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOpReadVariableOpDmlp_ae_notransform_v0_1_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
,MLP_Ae_NoTransform_v0.1/output_layer/BiasAddBiasAdd5MLP_Ae_NoTransform_v0.1/output_layer/MatMul:product:0CMLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,MLP_Ae_NoTransform_v0.1/output_layer/SoftmaxSoftmax5MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity6MLP_Ae_NoTransform_v0.1/output_layer/Softmax:softmax:0>^MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp=^MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOp>^MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp=^MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOp;^MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOp:^MLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOp<^MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOp;^MLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::2v
9MLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOp9MLP_Ae_NoTransform_v0.1/input_layer/MatMul/ReadVariableOp2~
=MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp=MLP_Ae_NoTransform_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp2~
=MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp=MLP_Ae_NoTransform_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp2|
<MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOp<MLP_Ae_NoTransform_v0.1/hidden_layer_2/MatMul/ReadVariableOp2z
;MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOp;MLP_Ae_NoTransform_v0.1/output_layer/BiasAdd/ReadVariableOp2x
:MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOp:MLP_Ae_NoTransform_v0.1/input_layer/BiasAdd/ReadVariableOp2x
:MLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOp:MLP_Ae_NoTransform_v0.1/output_layer/MatMul/ReadVariableOp2|
<MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOp<MLP_Ae_NoTransform_v0.1/hidden_layer_1/MatMul/ReadVariableOp:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
�	
�
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208061

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
c
*__inference_dropout_1_layer_call_fn_208098

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������@*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
b
C__inference_dropout_layer_call_and_return_conditional_losses_208035

inputs
identity�Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:����������j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�	
�
G__inference_input_layer_layer_call_and_return_conditional_losses_208008

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
/__inference_hidden_layer_1_layer_call_fn_208068

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������@�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688

inputs
identity�Q
dropout/rateConst*
valueB
 *  �>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:��������� �
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:��������� �
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:��������� R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:��������� a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:��������� i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*&
_input_shapes
:��������� :& "
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_208050

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_207551

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_208093

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_207854
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-207843**
f%R#
!__inference__wrapped_model_207490*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
P
input_layer_input;
#serving_default_input_layer_input:0����������@
output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�,
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
*t&call_and_return_all_conditional_losses
u_default_save_signature
v__call__"�)
_tf_keras_sequential�){"class_name": "Sequential", "name": "MLP_Ae_NoTransform_v0.1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "MLP_Ae_NoTransform_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "MLP_Ae_NoTransform_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 128], "config": {"batch_input_shape": [null, 128], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 128], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
*}&call_and_return_all_conditional_losses
~__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
*&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
�

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
�
7iter
	8decay
9learning_rate
:momentum
;rho	rmsl	rmsm	rmsn	rmso	'rmsp	(rmsq	1rmsr	2rmss"
	optimizer
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
 "
trackable_list_wrapper
�

	variables
trainable_variables
<metrics
=layer_regularization_losses

>layers
?non_trainable_variables
regularization_losses
v__call__
u_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
trainable_variables
@metrics
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
&:$
��2input_layer/kernel
:�2input_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
trainable_variables
Dmetrics
Elayer_regularization_losses

Flayers
Gnon_trainable_variables
regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
Knon_trainable_variables
regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
(:&	�@2hidden_layer_1/kernel
!:@2hidden_layer_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
 trainable_variables
Lmetrics
Mlayer_regularization_losses

Nlayers
Onon_trainable_variables
!regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
#	variables
$trainable_variables
Pmetrics
Qlayer_regularization_losses

Rlayers
Snon_trainable_variables
%regularization_losses
�__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
':%@ 2hidden_layer_2/kernel
!: 2hidden_layer_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)	variables
*trainable_variables
Tmetrics
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
+regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
-	variables
.trainable_variables
Xmetrics
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
/regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
%:# 2output_layer/kernel
:2output_layer/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
3	variables
4trainable_variables
\metrics
]layer_regularization_losses

^layers
_non_trainable_variables
5regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	atotal
	bcount
c
_fn_kwargs
d	variables
etrainable_variables
fregularization_losses
g	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
d	variables
etrainable_variables
hmetrics
ilayer_regularization_losses

jlayers
knon_trainable_variables
fregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
0:.
��2RMSprop/input_layer/kernel/rms
):'�2RMSprop/input_layer/bias/rms
2:0	�@2!RMSprop/hidden_layer_1/kernel/rms
+:)@2RMSprop/hidden_layer_1/bias/rms
1:/@ 2!RMSprop/hidden_layer_2/kernel/rms
+:) 2RMSprop/hidden_layer_2/bias/rms
/:- 2RMSprop/output_layer/kernel/rms
):'2RMSprop/output_layer/bias/rms
�2�
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207936
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207741
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207971
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207762�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_207490�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *1�.
,�)
input_layer_input����������
�2�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207831
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207997
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207796
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207984�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
G__inference_input_layer_layer_call_and_return_conditional_losses_208008�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_input_layer_layer_call_fn_208015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dropout_layer_call_and_return_conditional_losses_208035
C__inference_dropout_layer_call_and_return_conditional_losses_208040�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dropout_layer_call_fn_208050
(__inference_dropout_layer_call_fn_208045�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208061�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_hidden_layer_1_layer_call_fn_208068�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dropout_1_layer_call_and_return_conditional_losses_208088
E__inference_dropout_1_layer_call_and_return_conditional_losses_208093�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dropout_1_layer_call_fn_208098
*__inference_dropout_1_layer_call_fn_208103�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_hidden_layer_2_layer_call_fn_208121�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dropout_2_layer_call_and_return_conditional_losses_208141
E__inference_dropout_2_layer_call_and_return_conditional_losses_208146�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dropout_2_layer_call_fn_208156
*__inference_dropout_2_layer_call_fn_208151�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_output_layer_layer_call_and_return_conditional_losses_208167�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_output_layer_layer_call_fn_208174�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
=B;
$__inference_signature_wrapper_207854input_layer_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
/__inference_hidden_layer_2_layer_call_fn_208121O'(/�,
%�"
 �
inputs���������@
� "���������� �
,__inference_input_layer_layer_call_fn_208015Q0�-
&�#
!�
inputs����������
� "������������
G__inference_input_layer_layer_call_and_return_conditional_losses_208008^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208061]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
/__inference_hidden_layer_1_layer_call_fn_208068P0�-
&�#
!�
inputs����������
� "����������@�
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208114\'(/�,
%�"
 �
inputs���������@
� "%�"
�
0��������� 
� �
-__inference_output_layer_layer_call_fn_208174O12/�,
%�"
 �
inputs��������� 
� "�����������
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207936k'(128�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
!__inference__wrapped_model_207490�'(12;�8
1�.
,�)
input_layer_input����������
� ";�8
6
output_layer&�#
output_layer���������}
*__inference_dropout_1_layer_call_fn_208103O3�0
)�&
 �
inputs���������@
p 
� "����������@�
E__inference_dropout_2_layer_call_and_return_conditional_losses_208141\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
C__inference_dropout_layer_call_and_return_conditional_losses_208040^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_208035^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207984^'(128�5
.�+
!�
inputs����������
p

 
� "�����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_208088\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_208093\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
E__inference_dropout_2_layer_call_and_return_conditional_losses_208146\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207741v'(12C�@
9�6
,�)
input_layer_input����������
p

 
� "%�"
�
0���������
� }
(__inference_dropout_layer_call_fn_208050Q4�1
*�'
!�
inputs����������
p 
� "�����������}
(__inference_dropout_layer_call_fn_208045Q4�1
*�'
!�
inputs����������
p
� "������������
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207971k'(128�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� }
*__inference_dropout_2_layer_call_fn_208151O3�0
)�&
 �
inputs��������� 
p
� "���������� �
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207997^'(128�5
.�+
!�
inputs����������
p 

 
� "�����������
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207831i'(12C�@
9�6
,�)
input_layer_input����������
p 

 
� "����������}
*__inference_dropout_2_layer_call_fn_208156O3�0
)�&
 �
inputs��������� 
p 
� "���������� �
S__inference_MLP_Ae_NoTransform_v0.1_layer_call_and_return_conditional_losses_207762v'(12C�@
9�6
,�)
input_layer_input����������
p 

 
� "%�"
�
0���������
� �
H__inference_output_layer_layer_call_and_return_conditional_losses_208167\12/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� }
*__inference_dropout_1_layer_call_fn_208098O3�0
)�&
 �
inputs���������@
p
� "����������@�
8__inference_MLP_Ae_NoTransform_v0.1_layer_call_fn_207796i'(12C�@
9�6
,�)
input_layer_input����������
p

 
� "�����������
$__inference_signature_wrapper_207854�'(12P�M
� 
F�C
A
input_layer_input,�)
input_layer_input����������";�8
6
output_layer&�#
output_layer���������