ь
Ј§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d388Ё

input_layer/kernelVarHandleOp*
shape:
*#
shared_nameinput_layer/kernel*
dtype0*
_output_shapes
: 
{
&input_layer/kernel/Read/ReadVariableOpReadVariableOpinput_layer/kernel*
dtype0* 
_output_shapes
:

y
input_layer/biasVarHandleOp*!
shared_nameinput_layer/bias*
dtype0*
_output_shapes
: *
shape:
r
$input_layer/bias/Read/ReadVariableOpReadVariableOpinput_layer/bias*
dtype0*
_output_shapes	
:

hidden_layer_1/kernelVarHandleOp*
shape:	@*&
shared_namehidden_layer_1/kernel*
dtype0*
_output_shapes
: 

)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
dtype0*
_output_shapes
:	@
~
hidden_layer_1/biasVarHandleOp*$
shared_namehidden_layer_1/bias*
dtype0*
_output_shapes
: *
shape:@
w
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
dtype0*
_output_shapes
:@

hidden_layer_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@ *&
shared_namehidden_layer_2/kernel

)hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel*
dtype0*
_output_shapes

:@ 
~
hidden_layer_2/biasVarHandleOp*$
shared_namehidden_layer_2/bias*
dtype0*
_output_shapes
: *
shape: 
w
'hidden_layer_2/bias/Read/ReadVariableOpReadVariableOphidden_layer_2/bias*
dtype0*
_output_shapes
: 

output_layer/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
: *$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
dtype0*
_output_shapes

: 
z
output_layer/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
dtype0*
_output_shapes
:
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
shape: *
shared_nameRMSprop/decay*
dtype0*
_output_shapes
: 
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
dtype0*
_output_shapes
: 
~
RMSprop/learning_rateVarHandleOp*
shape: *&
shared_nameRMSprop/learning_rate*
dtype0*
_output_shapes
: 
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
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

RMSprop/input_layer/kernel/rmsVarHandleOp*
shape:
*/
shared_name RMSprop/input_layer/kernel/rms*
dtype0*
_output_shapes
: 

2RMSprop/input_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/kernel/rms*
dtype0* 
_output_shapes
:


RMSprop/input_layer/bias/rmsVarHandleOp*-
shared_nameRMSprop/input_layer/bias/rms*
dtype0*
_output_shapes
: *
shape:

0RMSprop/input_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/bias/rms*
dtype0*
_output_shapes	
:

!RMSprop/hidden_layer_1/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@*2
shared_name#!RMSprop/hidden_layer_1/kernel/rms

5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_1/kernel/rms*
dtype0*
_output_shapes
:	@

RMSprop/hidden_layer_1/bias/rmsVarHandleOp*0
shared_name!RMSprop/hidden_layer_1/bias/rms*
dtype0*
_output_shapes
: *
shape:@

3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_1/bias/rms*
dtype0*
_output_shapes
:@

!RMSprop/hidden_layer_2/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@ *2
shared_name#!RMSprop/hidden_layer_2/kernel/rms

5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_2/kernel/rms*
dtype0*
_output_shapes

:@ 

RMSprop/hidden_layer_2/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: *0
shared_name!RMSprop/hidden_layer_2/bias/rms

3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_2/bias/rms*
dtype0*
_output_shapes
: 

RMSprop/output_layer/kernel/rmsVarHandleOp*
shape
: *0
shared_name!RMSprop/output_layer/kernel/rms*
dtype0*
_output_shapes
: 

3RMSprop/output_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/kernel/rms*
dtype0*
_output_shapes

: 

RMSprop/output_layer/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_nameRMSprop/output_layer/bias/rms

1RMSprop/output_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/bias/rms*
dtype0*
_output_shapes
:

NoOpNoOp
Ј.
ConstConst"/device:CPU:0*у-
valueй-Bж- BЯ-
С
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
regularization_losses
trainable_variables
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api

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
 
8
0
1
2
3
'4
(5
16
27

<layer_regularization_losses
=non_trainable_variables

	variables
regularization_losses
>metrics
trainable_variables

?layers
 
 
 
 

trainable_variables
@layer_regularization_losses
Anon_trainable_variables
	variables
Bmetrics
regularization_losses

Clayers
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

trainable_variables
Dlayer_regularization_losses
Enon_trainable_variables
	variables
Fmetrics
regularization_losses

Glayers
 
 
 

trainable_variables
Hlayer_regularization_losses
Inon_trainable_variables
	variables
Jmetrics
regularization_losses

Klayers
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

trainable_variables
Llayer_regularization_losses
Mnon_trainable_variables
 	variables
Nmetrics
!regularization_losses

Olayers
 
 
 

#trainable_variables
Player_regularization_losses
Qnon_trainable_variables
$	variables
Rmetrics
%regularization_losses

Slayers
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

)trainable_variables
Tlayer_regularization_losses
Unon_trainable_variables
*	variables
Vmetrics
+regularization_losses

Wlayers
 
 
 

-trainable_variables
Xlayer_regularization_losses
Ynon_trainable_variables
.	variables
Zmetrics
/regularization_losses

[layers
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

3trainable_variables
\layer_regularization_losses
]non_trainable_variables
4	variables
^metrics
5regularization_losses

_layers
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
 
 

`0
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
x
	atotal
	bcount
c
_fn_kwargs
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

a0
b1
 

dtrainable_variables
hlayer_regularization_losses
inon_trainable_variables
e	variables
jmetrics
fregularization_losses

klayers
 

a0
b1
 
 

VARIABLE_VALUERMSprop/input_layer/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/input_layer/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!RMSprop/hidden_layer_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/hidden_layer_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!RMSprop/hidden_layer_2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/hidden_layer_2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/output_layer/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/output_layer/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

!serving_default_input_layer_inputPlaceholder*
shape:џџџџџџџџџ*
dtype0*(
_output_shapes
:џџџџџџџџџ
ж
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/bias*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-164448*-
f(R&
$__inference_signature_wrapper_164078*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
љ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input_layer/kernel/Read/ReadVariableOp$input_layer/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp)hidden_layer_2/kernel/Read/ReadVariableOp'hidden_layer_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2RMSprop/input_layer/kernel/rms/Read/ReadVariableOp0RMSprop/input_layer/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOp3RMSprop/output_layer/kernel/rms/Read/ReadVariableOp1RMSprop/output_layer/bias/rms/Read/ReadVariableOpConst*(
f#R!
__inference__traced_save_164492*
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
: *-
_gradient_op_typePartitionedCall-164493
Ј
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/input_layer/kernel/rmsRMSprop/input_layer/bias/rms!RMSprop/hidden_layer_1/kernel/rmsRMSprop/hidden_layer_1/bias/rms!RMSprop/hidden_layer_2/kernel/rmsRMSprop/hidden_layer_2/bias/rmsRMSprop/output_layer/kernel/rmsRMSprop/output_layer/bias/rms*-
_gradient_op_typePartitionedCall-164575*+
f&R$
"__inference__traced_restore_164574*
Tout
2*-
config_proto

GPU

CPU2*0J 8*#
Tin
2*
_output_shapes
: а
и	
у
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ@*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
и	
у
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_164285

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ@*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ@*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ@*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ь


4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164221

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-164044*X
fSRQ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164043*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2	
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
у
Ў
-__inference_output_layer_layer_call_fn_164398

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2*-
_gradient_op_typePartitionedCall-163953*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_163947*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
К
F
*__inference_dropout_2_layer_call_fn_164380

inputs
identity
PartitionedCallPartitionedCallinputs*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163919*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ *
Tin
2*-
_gradient_op_typePartitionedCall-163931`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
ў4
и

__inference__traced_save_164492
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

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_1285b5d112b944299990913465649611/parts

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
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Х
valueЛBИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
SaveV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:В

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_rmsprop_input_layer_kernel_rms_read_readvariableop7savev2_rmsprop_input_layer_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop8savev2_rmsprop_output_layer_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0
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

identity_1Identity_1:output:0*Џ
_input_shapes
: :
::	@:@:@ : : :: : : : : : : :
::	@:@:@ : : :: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : 
э


4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164020
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*'
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-164009*X
fSRQ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164008*
Tout
2*-
config_proto

GPU

CPU2*0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :1 -
+
_user_specified_nameinput_layer_input: : : : : 
!
І
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163986
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityЂ&hidden_layer_1/StatefulPartitionedCallЂ&hidden_layer_2/StatefulPartitionedCallЂ#input_layer/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCallЂ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-163737*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_163731*
Tout
2Ы
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-163787*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163775*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2М
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163809*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2б
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2*-
_gradient_op_typePartitionedCall-163859*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163847*
Tout
2О
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163881*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875*
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
:џџџџџџџџџ б
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163919*
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
:џџџџџџџџџ *-
_gradient_op_typePartitionedCall-163931Ж
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-163953*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_163947*
Tout
2
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall: : : : : : : :1 -
+
_user_specified_nameinput_layer_input: 

a
C__inference_dropout_layer_call_and_return_conditional_losses_163775

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
А
b
C__inference_dropout_layer_call_and_return_conditional_losses_163768

inputs
identityQ
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:џџџџџџџџџ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:џџџџџџџџџj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Љ
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_163912

inputs
identityQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *  >C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ 
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ђ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:џџџџџџџџџ *
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:џџџџџџџџџ *

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:џџџџџџџџџ *
T0"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
л	
р
G__inference_input_layer_layer_call_and_return_conditional_losses_163731

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_163919

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*'
_output_shapes
:џџџџџџџџџ *
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
ж	
у
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ  
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ *
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Љ
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_163840

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ@
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ђ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:џџџџџџџџџ@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs

a
C__inference_dropout_layer_call_and_return_conditional_losses_164264

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Љ
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_164312

inputs
identityQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *  >C
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ@
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ђ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:џџџџџџџџџ@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
В%

O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164008

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ&hidden_layer_1/StatefulPartitionedCallЂ&hidden_layer_2/StatefulPartitionedCallЂ#input_layer/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*-
_gradient_op_typePartitionedCall-163737*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_163731*
Tout
2л
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-163779*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163768*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2Ф
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
:џџџџџџџџџ@*-
_gradient_op_typePartitionedCall-163809*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ@*-
_gradient_op_typePartitionedCall-163851*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163840*
Tout
2Ц
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ *-
_gradient_op_typePartitionedCall-163881*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875*
Tout
2
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*'
_output_shapes
:џџџџџџџџџ *-
_gradient_op_typePartitionedCall-163923*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163912*
Tout
2*-
config_proto

GPU

CPU2*0J 8О
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163953*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_163947*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2ў
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : 

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_164370

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*'
_output_shapes
:џџџџџџџџџ *
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
А
b
C__inference_dropout_layer_call_and_return_conditional_losses_164259

inputs
identityQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?C
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:џџџџџџџџџ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:џџџџџџџџџj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
К
F
*__inference_dropout_1_layer_call_fn_164327

inputs
identity
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-163859*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163847*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Н
a
(__inference_dropout_layer_call_fn_164269

inputs
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-163779*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163768*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ч
А
/__inference_hidden_layer_2_layer_call_fn_164345

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ *
Tin
2*-
_gradient_op_typePartitionedCall-163881*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ *
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

c
E__inference_dropout_1_layer_call_and_return_conditional_losses_164317

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ш
А
/__inference_hidden_layer_1_layer_call_fn_164292

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163809*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ@*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Ь


4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164208

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*X
fSRQ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164008*
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
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-164009
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
ф
­
,__inference_input_layer_layer_call_fn_164239

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*-
_gradient_op_typePartitionedCall-163737*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_163731
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
О
c
*__inference_dropout_1_layer_call_fn_164322

inputs
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ@*-
_gradient_op_typePartitionedCall-163851*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163840*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ф:
Я
!__inference__wrapped_model_163714
input_layer_inputB
>mlp_gs_noshift_v0_1_input_layer_matmul_readvariableop_resourceC
?mlp_gs_noshift_v0_1_input_layer_biasadd_readvariableop_resourceE
Amlp_gs_noshift_v0_1_hidden_layer_1_matmul_readvariableop_resourceF
Bmlp_gs_noshift_v0_1_hidden_layer_1_biasadd_readvariableop_resourceE
Amlp_gs_noshift_v0_1_hidden_layer_2_matmul_readvariableop_resourceF
Bmlp_gs_noshift_v0_1_hidden_layer_2_biasadd_readvariableop_resourceC
?mlp_gs_noshift_v0_1_output_layer_matmul_readvariableop_resourceD
@mlp_gs_noshift_v0_1_output_layer_biasadd_readvariableop_resource
identityЂ9MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOpЂ8MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOpЂ9MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOpЂ8MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOpЂ6MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOpЂ5MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOpЂ7MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOpЂ6MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOpф
5MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOpReadVariableOp>mlp_gs_noshift_v0_1_input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Е
&MLP_GS_NoShift_v0.1/input_layer/MatMulMatMulinput_layer_input=MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџс
6MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOpReadVariableOp?mlp_gs_noshift_v0_1_input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:з
'MLP_GS_NoShift_v0.1/input_layer/BiasAddBiasAdd0MLP_GS_NoShift_v0.1/input_layer/MatMul:product:0>MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$MLP_GS_NoShift_v0.1/input_layer/ReluRelu0MLP_GS_NoShift_v0.1/input_layer/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
$MLP_GS_NoShift_v0.1/dropout/IdentityIdentity2MLP_GS_NoShift_v0.1/input_layer/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџщ
8MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOpReadVariableOpAmlp_gs_noshift_v0_1_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	@ж
)MLP_GS_NoShift_v0.1/hidden_layer_1/MatMulMatMul-MLP_GS_NoShift_v0.1/dropout/Identity:output:0@MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@ц
9MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOpBmlp_gs_noshift_v0_1_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@п
*MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAddBiasAdd3MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul:product:0AMLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
'MLP_GS_NoShift_v0.1/hidden_layer_1/ReluRelu3MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&MLP_GS_NoShift_v0.1/dropout_1/IdentityIdentity5MLP_GS_NoShift_v0.1/hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@ш
8MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOpReadVariableOpAmlp_gs_noshift_v0_1_hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ и
)MLP_GS_NoShift_v0.1/hidden_layer_2/MatMulMatMul/MLP_GS_NoShift_v0.1/dropout_1/Identity:output:0@MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ц
9MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOpBmlp_gs_noshift_v0_1_hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: п
*MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAddBiasAdd3MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul:product:0AMLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'MLP_GS_NoShift_v0.1/hidden_layer_2/ReluRelu3MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&MLP_GS_NoShift_v0.1/dropout_2/IdentityIdentity5MLP_GS_NoShift_v0.1/hidden_layer_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ ф
6MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOpReadVariableOp?mlp_gs_noshift_v0_1_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: д
'MLP_GS_NoShift_v0.1/output_layer/MatMulMatMul/MLP_GS_NoShift_v0.1/dropout_2/Identity:output:0>MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџт
7MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOpReadVariableOp@mlp_gs_noshift_v0_1_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:й
(MLP_GS_NoShift_v0.1/output_layer/BiasAddBiasAdd1MLP_GS_NoShift_v0.1/output_layer/MatMul:product:0?MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(MLP_GS_NoShift_v0.1/output_layer/SoftmaxSoftmax1MLP_GS_NoShift_v0.1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЬ
IdentityIdentity2MLP_GS_NoShift_v0.1/output_layer/Softmax:softmax:0:^MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp9^MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOp:^MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp9^MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOp7^MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOp6^MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOp8^MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOp7^MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2t
8MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOp8MLP_GS_NoShift_v0.1/hidden_layer_2/MatMul/ReadVariableOp2p
6MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOp6MLP_GS_NoShift_v0.1/input_layer/BiasAdd/ReadVariableOp2n
5MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOp5MLP_GS_NoShift_v0.1/input_layer/MatMul/ReadVariableOp2v
9MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp9MLP_GS_NoShift_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp2p
6MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOp6MLP_GS_NoShift_v0.1/output_layer/MatMul/ReadVariableOp2t
8MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOp8MLP_GS_NoShift_v0.1/hidden_layer_1/MatMul/ReadVariableOp2v
9MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp9MLP_GS_NoShift_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp2r
7MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOp7MLP_GS_NoShift_v0.1/output_layer/BiasAdd/ReadVariableOp:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
л	
р
G__inference_input_layer_layer_call_and_return_conditional_losses_164232

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ђ+
В
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164195

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityЂ%hidden_layer_1/BiasAdd/ReadVariableOpЂ$hidden_layer_1/MatMul/ReadVariableOpЂ%hidden_layer_2/BiasAdd/ReadVariableOpЂ$hidden_layer_2/MatMul/ReadVariableOpЂ"input_layer/BiasAdd/ReadVariableOpЂ!input_layer/MatMul/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpМ
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:

input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџo
dropout/IdentityIdentityinput_layer/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџС
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	@
hidden_layer_1/MatMulMatMuldropout/Identity:output:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@О
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ѓ
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ@*
T0s
dropout_1/IdentityIdentity!hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@Р
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ 
hidden_layer_2/MatMulMatMuldropout_1/Identity:output:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ *
T0О
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ѓ
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ s
dropout_2/IdentityIdentity!hidden_layer_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ М
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: 
output_layer/MatMulMatMuldropout_2/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџК
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : : : : : 
Џ


$__inference_signature_wrapper_164078
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
config_proto

GPU

CPU2*0J 8*
Tin
2	*'
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-164067**
f%R#
!__inference__wrapped_model_163714*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: : : 
й	
с
H__inference_output_layer_layer_call_and_return_conditional_losses_163947

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Я[
В
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164160

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityЂ%hidden_layer_1/BiasAdd/ReadVariableOpЂ$hidden_layer_1/MatMul/ReadVariableOpЂ%hidden_layer_2/BiasAdd/ReadVariableOpЂ$hidden_layer_2/MatMul/ReadVariableOpЂ"input_layer/BiasAdd/ReadVariableOpЂ!input_layer/MatMul/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpМ
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:

input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0i
input_layer/ReluReluinput_layer/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
T0Y
dropout/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?c
dropout/dropout/ShapeShapeinput_layer/Relu:activations:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџЄ
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Л
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*(
_output_shapes
:џџџџџџџџџ*
T0­
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
dropout/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0Ђ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/dropout/mulMulinput_layer/Relu:activations:0dropout/dropout/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:џџџџџџџџџ
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџС
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	@
hidden_layer_1/MatMulMatMuldropout/dropout/mul_1:z:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ@*
T0О
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ѓ
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
dropout_1/dropout/rateConst*
valueB
 *  >*
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
$dropout_1/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ? 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ@Њ
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Р
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*'
_output_shapes
:џџџџџџџџџ@*
T0В
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*'
_output_shapes
:џџџџџџџџџ@*
T0\
dropout_1/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
_output_shapes
: *
T0Ї
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dropout_1/dropout/mulMul!hidden_layer_1/Relu:activations:0dropout_1/dropout/truediv:z:0*'
_output_shapes
:џџџџџџџџџ@*
T0
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:џџџџџџџџџ@
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Р
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ 
hidden_layer_2/MatMulMatMuldropout_1/dropout/mul_1:z:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ *
T0О
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ѓ
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
dropout_2/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: h
dropout_2/dropout/ShapeShape!hidden_layer_2/Relu:activations:0*
_output_shapes
:*
T0i
$dropout_2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ? 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ Њ
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Р
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*'
_output_shapes
:џџџџџџџџџ *
T0В
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
dropout_2/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_2/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: Ї
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dropout_2/dropout/mulMul!hidden_layer_2/Relu:activations:0dropout_2/dropout/truediv:z:0*'
_output_shapes
:џџџџџџџџџ *
T0
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:џџџџџџџџџ 
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*'
_output_shapes
:џџџџџџџџџ *
T0М
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: 
output_layer/MatMulMatMuldropout_2/dropout/mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0К
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
І\
њ
"__inference__traced_restore_164574
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
identity_24ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*Х
valueЛBИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
RestoreV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	L
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
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0	
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0*
dtype0	*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
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
:
AssignVariableOp_15AssignVariableOp2assignvariableop_15_rmsprop_input_layer_kernel_rmsIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp0assignvariableop_16_rmsprop_input_layer_bias_rmsIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rmsIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp3assignvariableop_18_rmsprop_hidden_layer_1_bias_rmsIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rmsIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp3assignvariableop_20_rmsprop_hidden_layer_2_bias_rmsIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0
AssignVariableOp_21AssignVariableOp3assignvariableop_21_rmsprop_output_layer_kernel_rmsIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp1assignvariableop_22_rmsprop_output_layer_bias_rmsIdentity_22:output:0*
dtype0*
_output_shapes
 
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
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Щ
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ж
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
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
AssignVariableOpAssignVariableOp: : : :	 :
 : : : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : 
г%

O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163965
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ&hidden_layer_1/StatefulPartitionedCallЂ&hidden_layer_2/StatefulPartitionedCallЂ#input_layer/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCallЂ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-163737*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_163731*
Tout
2*-
config_proto

GPU

CPU2*0J 8л
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-163779*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163768*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ф
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803*
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
:џџџџџџџџџ@*-
_gradient_op_typePartitionedCall-163809
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163840*
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
:џџџџџџџџџ@*-
_gradient_op_typePartitionedCall-163851Ц
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163881*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875*
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
:џџџџџџџџџ 
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-163923*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163912*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ *
Tin
2О
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
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-163953*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_163947ў
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: : : : : : : :1 -
+
_user_specified_nameinput_layer_input: 

c
E__inference_dropout_1_layer_call_and_return_conditional_losses_163847

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ќ 

O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164043

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityЂ&hidden_layer_1/StatefulPartitionedCallЂ&hidden_layer_2/StatefulPartitionedCallЂ#input_layer/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163737*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_163731*
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
:џџџџџџџџџЫ
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-163787*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163775*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2М
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-163809*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_163803*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2б
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-163859*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_163847*
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
:џџџџџџџџџ@О
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ *
Tin
2*-
_gradient_op_typePartitionedCall-163881*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_163875*
Tout
2*-
config_proto

GPU

CPU2*0J 8б
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-163931*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163919*
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
:џџџџџџџџџ Ж
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ*
Tin
2*-
_gradient_op_typePartitionedCall-163953*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_163947*
Tout
2*-
config_proto

GPU

CPU2*0J 8
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
Й
D
(__inference_dropout_layer_call_fn_164274

inputs
identity
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-163787*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_163775*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
э


4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164055
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-164044*X
fSRQ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164043*
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
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :1 -
+
_user_specified_nameinput_layer_input: : : : : 
й	
с
H__inference_output_layer_layer_call_and_return_conditional_losses_164391

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
О
c
*__inference_dropout_2_layer_call_fn_164375

inputs
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputs*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_163912*
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
:џџџџџџџџџ *-
_gradient_op_typePartitionedCall-163923
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ж	
у
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_164338

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ  
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ *
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Љ
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_164365

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*'
_output_shapes
:џџџџџџџџџ *
T0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ђ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ R
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:џџџџџџџџџ i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Ф
serving_defaultА
P
input_layer_input;
#serving_default_input_layer_input:0џџџџџџџџџ@
output_layer0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Пр
н,
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
regularization_losses
trainable_variables
	keras_api

signatures
t_default_save_signature
*u&call_and_return_all_conditional_losses
v__call__"Т)
_tf_keras_sequentialЃ){"class_name": "Sequential", "name": "MLP_GS_NoShift_v0.1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "MLP_GS_NoShift_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "MLP_GS_NoShift_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Й
trainable_variables
	variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Њ
_tf_keras_layer{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 128], "config": {"batch_input_shape": [null, 128], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
Є

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"џ
_tf_keras_layerх{"class_name": "Dense", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 128], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 128], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
Ћ
trainable_variables
	variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}


kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
*}&call_and_return_all_conditional_losses
~__call__"л
_tf_keras_layerС{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
Б
#trainable_variables
$	variables
%regularization_losses
&	keras_api
*&call_and_return_all_conditional_losses
__call__"Ё
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+&call_and_return_all_conditional_losses
__call__"к
_tf_keras_layerР{"class_name": "Dense", "name": "hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
В
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+&call_and_return_all_conditional_losses
__call__"Ё
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
џ

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+&call_and_return_all_conditional_losses
__call__"и
_tf_keras_layerО{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
Њ
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
 "
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
З
<layer_regularization_losses
=non_trainable_variables

	variables
regularization_losses
>metrics
trainable_variables

?layers
v__call__
t_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

trainable_variables
@layer_regularization_losses
Anon_trainable_variables
	variables
Bmetrics
regularization_losses

Clayers
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
&:$
2input_layer/kernel
:2input_layer/bias
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

trainable_variables
Dlayer_regularization_losses
Enon_trainable_variables
	variables
Fmetrics
regularization_losses

Glayers
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

trainable_variables
Hlayer_regularization_losses
Inon_trainable_variables
	variables
Jmetrics
regularization_losses

Klayers
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
(:&	@2hidden_layer_1/kernel
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

trainable_variables
Llayer_regularization_losses
Mnon_trainable_variables
 	variables
Nmetrics
!regularization_losses

Olayers
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

#trainable_variables
Player_regularization_losses
Qnon_trainable_variables
$	variables
Rmetrics
%regularization_losses

Slayers
__call__
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

)trainable_variables
Tlayer_regularization_losses
Unon_trainable_variables
*	variables
Vmetrics
+regularization_losses

Wlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

-trainable_variables
Xlayer_regularization_losses
Ynon_trainable_variables
.	variables
Zmetrics
/regularization_losses

[layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:# 2output_layer/kernel
:2output_layer/bias
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

3trainable_variables
\layer_regularization_losses
]non_trainable_variables
4	variables
^metrics
5regularization_losses

_layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
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

	atotal
	bcount
c
_fn_kwargs
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
+&call_and_return_all_conditional_losses
__call__"х
_tf_keras_layerЫ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper

dtrainable_variables
hlayer_regularization_losses
inon_trainable_variables
e	variables
jmetrics
fregularization_losses

klayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:.
2RMSprop/input_layer/kernel/rms
):'2RMSprop/input_layer/bias/rms
2:0	@2!RMSprop/hidden_layer_1/kernel/rms
+:)@2RMSprop/hidden_layer_1/bias/rms
1:/@ 2!RMSprop/hidden_layer_2/kernel/rms
+:) 2RMSprop/hidden_layer_2/bias/rms
/:- 2RMSprop/output_layer/kernel/rms
):'2RMSprop/output_layer/bias/rms
ъ2ч
!__inference__wrapped_model_163714С
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *1Ђ.
,)
input_layer_inputџџџџџџџџџ
2
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164195
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163965
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164160
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163986Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164055
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164020
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164208
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164221Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ё2ю
G__inference_input_layer_layer_call_and_return_conditional_losses_164232Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_input_layer_layer_call_fn_164239Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ф2С
C__inference_dropout_layer_call_and_return_conditional_losses_164259
C__inference_dropout_layer_call_and_return_conditional_losses_164264Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_dropout_layer_call_fn_164274
(__inference_dropout_layer_call_fn_164269Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
є2ё
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_164285Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_hidden_layer_1_layer_call_fn_164292Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_1_layer_call_and_return_conditional_losses_164317
E__inference_dropout_1_layer_call_and_return_conditional_losses_164312Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_1_layer_call_fn_164322
*__inference_dropout_1_layer_call_fn_164327Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
є2ё
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_164338Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_hidden_layer_2_layer_call_fn_164345Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_2_layer_call_and_return_conditional_losses_164370
E__inference_dropout_2_layer_call_and_return_conditional_losses_164365Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_2_layer_call_fn_164375
*__inference_dropout_2_layer_call_fn_164380Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
H__inference_output_layer_layer_call_and_return_conditional_losses_164391Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_output_layer_layer_call_fn_164398Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
=B;
$__inference_signature_wrapper_164078input_layer_input
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_164259^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_164264^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 }
(__inference_dropout_layer_call_fn_164274Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ}
(__inference_dropout_layer_call_fn_164269Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ}
*__inference_dropout_2_layer_call_fn_164375O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "џџџџџџџџџ }
*__inference_dropout_2_layer_call_fn_164380O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "џџџџџџџџџ Щ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163965v'(12CЂ@
9Ђ6
,)
input_layer_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ё
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164055i'(12CЂ@
9Ђ6
,)
input_layer_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЈ
H__inference_output_layer_layer_call_and_return_conditional_losses_164391\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 О
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164195k'(128Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Щ
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_163986v'(12CЂ@
9Ђ6
,)
input_layer_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Т
$__inference_signature_wrapper_164078'(12PЂM
Ђ 
FЊC
A
input_layer_input,)
input_layer_inputџџџџџџџџџ";Њ8
6
output_layer&#
output_layerџџџџџџџџџЉ
G__inference_input_layer_layer_call_and_return_conditional_losses_164232^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
/__inference_hidden_layer_2_layer_call_fn_164345O'(/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ 
,__inference_input_layer_layer_call_fn_164239Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЊ
!__inference__wrapped_model_163714'(12;Ђ8
1Ђ.
,)
input_layer_inputџџџџџџџџџ
Њ ";Њ8
6
output_layer&#
output_layerџџџџџџџџџЅ
E__inference_dropout_1_layer_call_and_return_conditional_losses_164312\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 
/__inference_hidden_layer_1_layer_call_fn_164292P0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ћ
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_164285]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 Ѕ
E__inference_dropout_1_layer_call_and_return_conditional_losses_164317\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164208^'(128Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЊ
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_164338\'(/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164221^'(128Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_output_layer_layer_call_fn_164398O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ}
*__inference_dropout_1_layer_call_fn_164322O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@Ё
4__inference_MLP_GS_NoShift_v0.1_layer_call_fn_164020i'(12CЂ@
9Ђ6
,)
input_layer_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџО
O__inference_MLP_GS_NoShift_v0.1_layer_call_and_return_conditional_losses_164160k'(128Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dropout_1_layer_call_fn_164327O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@Ѕ
E__inference_dropout_2_layer_call_and_return_conditional_losses_164365\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "%Ђ"

0џџџџџџџџџ 
 Ѕ
E__inference_dropout_2_layer_call_and_return_conditional_losses_164370\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "%Ђ"

0џџџџџџџџџ 
 