Ен
®э
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d388пД
В
input_layer/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
А А*#
shared_nameinput_layer/kernel
{
&input_layer/kernel/Read/ReadVariableOpReadVariableOpinput_layer/kernel*
dtype0* 
_output_shapes
:
А А
y
input_layer/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*!
shared_nameinput_layer/bias
r
$input_layer/bias/Read/ReadVariableOpReadVariableOpinput_layer/bias*
dtype0*
_output_shapes	
:А
З
hidden_layer_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*&
shared_namehidden_layer_1/kernel
А
)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
dtype0*
_output_shapes
:	А@
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
Ж
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
В
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
RMSprop/momentumVarHandleOp*!
shared_nameRMSprop/momentum*
dtype0*
_output_shapes
: *
shape: 
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
dtype0*
_output_shapes
: 
j
RMSprop/rhoVarHandleOp*
shared_nameRMSprop/rho*
dtype0*
_output_shapes
: *
shape: 
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
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
Ъ
RMSprop/input_layer/kernel/rmsVarHandleOp*/
shared_name RMSprop/input_layer/kernel/rms*
dtype0*
_output_shapes
: *
shape:
А А
У
2RMSprop/input_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/kernel/rms*
dtype0* 
_output_shapes
:
А А
С
RMSprop/input_layer/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*-
shared_nameRMSprop/input_layer/bias/rms
К
0RMSprop/input_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/input_layer/bias/rms*
dtype0*
_output_shapes	
:А
Я
!RMSprop/hidden_layer_1/kernel/rmsVarHandleOp*
shape:	А@*2
shared_name#!RMSprop/hidden_layer_1/kernel/rms*
dtype0*
_output_shapes
: 
Ш
5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_1/kernel/rms*
dtype0*
_output_shapes
:	А@
Ц
RMSprop/hidden_layer_1/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*0
shared_name!RMSprop/hidden_layer_1/bias/rms
П
3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_1/bias/rms*
dtype0*
_output_shapes
:@
Ю
!RMSprop/hidden_layer_2/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@ *2
shared_name#!RMSprop/hidden_layer_2/kernel/rms
Ч
5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/hidden_layer_2/kernel/rms*
dtype0*
_output_shapes

:@ 
Ц
RMSprop/hidden_layer_2/bias/rmsVarHandleOp*
shape: *0
shared_name!RMSprop/hidden_layer_2/bias/rms*
dtype0*
_output_shapes
: 
П
3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/hidden_layer_2/bias/rms*
dtype0*
_output_shapes
: 
Ъ
RMSprop/output_layer/kernel/rmsVarHandleOp*
shape
: *0
shared_name!RMSprop/output_layer/kernel/rms*
dtype0*
_output_shapes
: 
У
3RMSprop/output_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/kernel/rms*
dtype0*
_output_shapes

: 
Т
RMSprop/output_layer/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_nameRMSprop/output_layer/bias/rms
Л
1RMSprop/output_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/bias/rms*
dtype0*
_output_shapes
:

NoOpNoOp
®.
ConstConst"/device:CPU:0*г-
valueў-B÷- Bѕ-
Ѕ
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

regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
Ч
7iter
	8decay
9learning_rate
:momentum
;rho	rmsl	rmsm	rmsn	rmso	'rmsp	(rmsq	1rmsr	2rmss
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
8
0
1
2
3
'4
(5
16
27
Ъ
<non_trainable_variables
=layer_regularization_losses

regularization_losses
trainable_variables
	variables

>layers
?metrics
 
 
 
 
Ъ
@non_trainable_variables
regularization_losses
Alayer_regularization_losses
trainable_variables
	variables

Blayers
Cmetrics
^\
VARIABLE_VALUEinput_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEinput_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
Dnon_trainable_variables
regularization_losses
Elayer_regularization_losses
trainable_variables
	variables

Flayers
Gmetrics
 
 
 
Ъ
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses
trainable_variables
	variables

Jlayers
Kmetrics
a_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
Lnon_trainable_variables
regularization_losses
Mlayer_regularization_losses
 trainable_variables
!	variables

Nlayers
Ometrics
 
 
 
Ъ
Pnon_trainable_variables
#regularization_losses
Qlayer_regularization_losses
$trainable_variables
%	variables

Rlayers
Smetrics
a_
VARIABLE_VALUEhidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
Ъ
Tnon_trainable_variables
)regularization_losses
Ulayer_regularization_losses
*trainable_variables
+	variables

Vlayers
Wmetrics
 
 
 
Ъ
Xnon_trainable_variables
-regularization_losses
Ylayer_regularization_losses
.trainable_variables
/	variables

Zlayers
[metrics
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
Ъ
\non_trainable_variables
3regularization_losses
]layer_regularization_losses
4trainable_variables
5	variables

^layers
_metrics
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
1
0
1
2
3
4
5
6

`0
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
dregularization_losses
etrainable_variables
f	variables
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

a0
b1
Ъ
hnon_trainable_variables
dregularization_losses
ilayer_regularization_losses
etrainable_variables
f	variables

jlayers
kmetrics

a0
b1
 
 
 
ЙЖ
VARIABLE_VALUERMSprop/input_layer/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUERMSprop/input_layer/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE!RMSprop/hidden_layer_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUERMSprop/hidden_layer_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE!RMSprop/hidden_layer_2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUERMSprop/hidden_layer_2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUERMSprop/output_layer/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUERMSprop/output_layer/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Ж
!serving_default_input_layer_inputPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€А *
shape:€€€€€€€€€А 
”
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/bias*-
_gradient_op_typePartitionedCall-208220*-
f(R&
$__inference_signature_wrapper_207850*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
ц	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input_layer/kernel/Read/ReadVariableOp$input_layer/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp)hidden_layer_2/kernel/Read/ReadVariableOp'hidden_layer_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2RMSprop/input_layer/kernel/rms/Read/ReadVariableOp0RMSprop/input_layer/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_1/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_1/bias/rms/Read/ReadVariableOp5RMSprop/hidden_layer_2/kernel/rms/Read/ReadVariableOp3RMSprop/hidden_layer_2/bias/rms/Read/ReadVariableOp3RMSprop/output_layer/kernel/rms/Read/ReadVariableOp1RMSprop/output_layer/bias/rms/Read/ReadVariableOpConst*(
f#R!
__inference__traced_save_208264*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *$
Tin
2	*-
_gradient_op_typePartitionedCall-208265
•
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_layer/kernelinput_layer/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biasoutput_layer/kerneloutput_layer/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/input_layer/kernel/rmsRMSprop/input_layer/bias/rms!RMSprop/hidden_layer_1/kernel/rmsRMSprop/hidden_layer_1/bias/rms!RMSprop/hidden_layer_2/kernel/rmsRMSprop/hidden_layer_2/bias/rmsRMSprop/output_layer/kernel/rmsRMSprop/output_layer/bias/rms*#
Tin
2*
_output_shapes
: *-
_gradient_op_typePartitionedCall-208347*+
f&R$
"__inference__traced_restore_208346*
Tout
2**
config_proto

GPU 

CPU2J 8ІТ
—[
і
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207932

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐ%hidden_layer_1/BiasAdd/ReadVariableOpҐ$hidden_layer_1/MatMul/ReadVariableOpҐ%hidden_layer_2/BiasAdd/ReadVariableOpҐ$hidden_layer_2/MatMul/ReadVariableOpҐ"input_layer/BiasAdd/ReadVariableOpҐ!input_layer/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpЉ
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А АВ
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ає
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АЫ
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аi
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
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
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Э
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*(
_output_shapes
:€€€€€€€€€А*
T0§
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ї
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А≠
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
dropout/dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: А
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: Ґ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*(
_output_shapes
:€€€€€€€€€АК
dropout/dropout/mulMulinput_layer/Relu:activations:0dropout/dropout/truediv:z:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:€€€€€€€€€АВ
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АЅ
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@Ъ
hidden_layer_1/MatMulMatMuldropout/dropout/mul_1:z:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€@*
T0[
dropout_1/dropout/rateConst*
valueB
 *  А>*
dtype0*
_output_shapes
: h
dropout_1/dropout/ShapeShape!hidden_layer_1/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: †
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€@™
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ј
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
dropout_1/dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: А
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ж
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: І
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Р
dropout_1/dropout/mulMul!hidden_layer_1/Relu:activations:0dropout_1/dropout/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€@З
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@ј
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ Ь
hidden_layer_2/MatMulMatMuldropout_1/dropout/mul_1:z:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Њ
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: £
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€ *
T0[
dropout_2/dropout/rateConst*
valueB
 *  А>*
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
 *  А?*
dtype0*
_output_shapes
: †
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€ ™
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ј
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ≤
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \
dropout_2/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?А
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_2/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ж
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: І
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
dropout_2/dropout/mulMul!hidden_layer_2/Relu:activations:0dropout_2/dropout/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:€€€€€€€€€ *

SrcT0
З
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*'
_output_shapes
:€€€€€€€€€ *
T0Љ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: Ш
output_layer/MatMulMatMuldropout_2/dropout/mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0Ї
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Э
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
џ	
а
G__inference_input_layer_layer_call_and_return_conditional_losses_207507

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:€€€€€€€€€А*
T0°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АМ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ў	
г
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208057

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
∞
b
C__inference_dropout_layer_call_and_return_conditional_losses_207544

inputs
identityИQ
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*(
_output_shapes
:€€€€€€€€€А*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: £
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:€€€€€€€€€А*
T0R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: К
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:€€€€€€€€€Аj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
ф+
і
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207967

inputs.
*input_layer_matmul_readvariableop_resource/
+input_layer_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐ%hidden_layer_1/BiasAdd/ReadVariableOpҐ$hidden_layer_1/MatMul/ReadVariableOpҐ%hidden_layer_2/BiasAdd/ReadVariableOpҐ$hidden_layer_2/MatMul/ReadVariableOpҐ"input_layer/BiasAdd/ReadVariableOpҐ!input_layer/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpЉ
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А АВ
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ає
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АЫ
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аi
input_layer/ReluReluinput_layer/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аo
dropout/IdentityIdentityinput_layer/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АЅ
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@Ъ
hidden_layer_1/MatMulMatMuldropout/Identity:output:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€@*
T0n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_1/IdentityIdentity!hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@ј
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ Ь
hidden_layer_2/MatMulMatMuldropout_1/Identity:output:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Њ
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: £
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ s
dropout_2/IdentityIdentity!hidden_layer_2/Relu:activations:0*'
_output_shapes
:€€€€€€€€€ *
T0Љ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: Ш
output_layer/MatMulMatMuldropout_2/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Э
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
Е
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:€€€€€€€€€@*
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:& "
 
_user_specified_nameinputs
о

Ъ
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207831
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-207820*Z
fURS
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207819*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :1 -
+
_user_specified_nameinput_layer_input: : : : : 
ґ
D
(__inference_dropout_layer_call_fn_208046

inputs
identityЪ
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€Аa
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
Ќ

П
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207993

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-207820*Z
fURS
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207819В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
©
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_208084

inputs
identityИQ
dropout/rateConst*
valueB
 *  А>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€@М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ґ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:€€€€€€€€€@*
T0Ф
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Й
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:€€€€€€€€€@*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:€€€€€€€€€@*
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:& "
 
_user_specified_nameinputs
Ј
F
*__inference_dropout_1_layer_call_fn_208099

inputs
identityЫ
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207635*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:& "
 
_user_specified_nameinputs
∞
b
C__inference_dropout_layer_call_and_return_conditional_losses_208031

inputs
identityИQ
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*(
_output_shapes
:€€€€€€€€€А*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0£
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:€€€€€€€€€АR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: К
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:€€€€€€€€€А*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
©
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616

inputs
identityИQ
dropout/rateConst*
valueB
 *  А>*
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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€@М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ґ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Й
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:& "
 
_user_specified_nameinputs
Ў	
г
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
о

Ъ
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207796
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-207785*Z
fURS
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207784*
Tout
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
ї
c
*__inference_dropout_1_layer_call_fn_208094

inputs
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputs*'
_output_shapes
:€€€€€€€€€@*
Tin
2*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616*
Tout
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Я%
З
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207784

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ&hidden_layer_1/StatefulPartitionedCallҐ&hidden_layer_2/StatefulPartitionedCallҐ#input_layer/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallФ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:€€€€€€€€€А*
Tin
2Ў
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€АЅ
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2А
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616*
Tout
2√
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ В
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ *-
_gradient_op_typePartitionedCall-207699*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688*
Tout
2ї
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2ю
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
Ќ

П
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207980

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-207785*Z
fURS
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207784*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
Ј
F
*__inference_dropout_2_layer_call_fn_208152

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ *-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :& "
 
_user_specified_nameinputs
ю4
Ў

__inference__traced_save_208264
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

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_3fb57ff1841e458e83f35bd07becaba7/part*
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
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ь
SaveV2/tensor_namesConst"/device:CPU:0*≈
valueїBЄB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Ы
SaveV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:≤

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_rmsprop_input_layer_kernel_rms_read_readvariableop7savev2_rmsprop_input_layer_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop8savev2_rmsprop_output_layer_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
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
B √
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2є
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
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

identity_1Identity_1:output:0*ѓ
_input_shapesЭ
Ъ: :
А А:А:	А@:@:@ : : :: : : : : : : :
А А:А:	А@:@:@ : : :: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : 
К!
®
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207762
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИҐ&hidden_layer_1/StatefulPartitionedCallҐ&hidden_layer_2/StatefulPartitionedCallҐ#input_layer/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallЯ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2**
config_proto

GPU 

CPU2J 8»
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551є
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*'
_output_shapes
:€€€€€€€€€@*
Tin
2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2**
config_proto

GPU 

CPU2J 8ќ
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207635*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@ї
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€ *
Tin
2*-
_gradient_op_typePartitionedCall-207657ќ
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ ≥
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-207729Ф
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall: : : : : : : :1 -
+
_user_specified_nameinput_layer_input: 
б
≠
,__inference_input_layer_layer_call_fn_208011

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€АГ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
÷	
г
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ †
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€ *
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:€€€€€€€€€ *
T0"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
©
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_208137

inputs
identityИQ
dropout/rateConst*
valueB
 *  А>*
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
 *  А?*
dtype0*
_output_shapes
: М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€ М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ґ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Й
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€ o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€ i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :& "
 
_user_specified_nameinputs
ў	
б
H__inference_output_layer_layer_call_and_return_conditional_losses_208163

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:€€€€€€€€€*
T0К
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ј%
Т
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207741
input_layer_input.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ&hidden_layer_1/StatefulPartitionedCallҐ&hidden_layer_2/StatefulPartitionedCallҐ#input_layer/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallЯ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_input*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€АЎ
dropout/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:€€€€€€€€€А*
Tin
2Ѕ
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€@*
Tin
2*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2А
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-207627*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207616*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@√
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€ *
Tin
2*-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2В
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ *-
_gradient_op_typePartitionedCall-207699*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688ї
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723ю
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : 
џ	
а
G__inference_input_layer_layer_call_and_return_conditional_losses_208004

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:€€€€€€€€€А*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
T0М
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ї
a
(__inference_dropout_layer_call_fn_208041

inputs
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-207555*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207544*
Tout
2Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
д
∞
/__inference_hidden_layer_2_layer_call_fn_208117

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€ *
Tin
2*-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ў	
б
H__inference_output_layer_layer_call_and_return_conditional_losses_207723

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Е
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :& "
 
_user_specified_nameinputs
Е
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_208089

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:€€€€€€€€€@*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:& "
 
_user_specified_nameinputs
¶\
ъ
"__inference__traced_restore_208346
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
identity_24ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1Я
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*≈
valueїBЄB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEЮ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B С
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

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0Г
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:И
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0Ж
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0И
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:Ж
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Ж
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Д
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

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0А
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Л
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Ж
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Б
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0{
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0{
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp2assignvariableop_15_rmsprop_input_layer_kernel_rmsIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Т
AssignVariableOp_16AssignVariableOp0assignvariableop_16_rmsprop_input_layer_bias_rmsIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Ч
AssignVariableOp_17AssignVariableOp5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rmsIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Х
AssignVariableOp_18AssignVariableOp3assignvariableop_18_rmsprop_hidden_layer_1_bias_rmsIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0Ч
AssignVariableOp_19AssignVariableOp5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rmsIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0Х
AssignVariableOp_20AssignVariableOp3assignvariableop_20_rmsprop_hidden_layer_2_bias_rmsIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:Х
AssignVariableOp_21AssignVariableOp3assignvariableop_21_rmsprop_output_layer_kernel_rmsIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp1assignvariableop_22_rmsprop_output_layer_bias_rmsIdentity_22:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 …
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ÷
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
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
AssignVariableOp_4AssignVariableOp_4: : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : 
А<
п
!__inference__wrapped_model_207490
input_layer_inputD
@mlp_ae_flattened_v0_1_input_layer_matmul_readvariableop_resourceE
Amlp_ae_flattened_v0_1_input_layer_biasadd_readvariableop_resourceG
Cmlp_ae_flattened_v0_1_hidden_layer_1_matmul_readvariableop_resourceH
Dmlp_ae_flattened_v0_1_hidden_layer_1_biasadd_readvariableop_resourceG
Cmlp_ae_flattened_v0_1_hidden_layer_2_matmul_readvariableop_resourceH
Dmlp_ae_flattened_v0_1_hidden_layer_2_biasadd_readvariableop_resourceE
Amlp_ae_flattened_v0_1_output_layer_matmul_readvariableop_resourceF
Bmlp_ae_flattened_v0_1_output_layer_biasadd_readvariableop_resource
identityИҐ;MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOpҐ:MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOpҐ;MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOpҐ:MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOpҐ8MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOpҐ7MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOpҐ9MLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOpҐ8MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOpи
7MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOpReadVariableOp@mlp_ae_flattened_v0_1_input_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А Ає
(MLP_Ae_Flattened_v0.1/input_layer/MatMulMatMulinput_layer_input?MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ае
8MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOpReadVariableOpAmlp_ae_flattened_v0_1_input_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АЁ
)MLP_Ae_Flattened_v0.1/input_layer/BiasAddBiasAdd2MLP_Ae_Flattened_v0.1/input_layer/MatMul:product:0@MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
&MLP_Ae_Flattened_v0.1/input_layer/ReluRelu2MLP_Ae_Flattened_v0.1/input_layer/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЫ
&MLP_Ae_Flattened_v0.1/dropout/IdentityIdentity4MLP_Ae_Flattened_v0.1/input_layer/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€Ан
:MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOpReadVariableOpCmlp_ae_flattened_v0_1_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@№
+MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMulMatMul/MLP_Ae_Flattened_v0.1/dropout/Identity:output:0BMLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@к
;MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOpDmlp_ae_flattened_v0_1_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@е
,MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAddBiasAdd5MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul:product:0CMLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€@*
T0Ъ
)MLP_Ae_Flattened_v0.1/hidden_layer_1/ReluRelu5MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Я
(MLP_Ae_Flattened_v0.1/dropout_1/IdentityIdentity7MLP_Ae_Flattened_v0.1/hidden_layer_1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@м
:MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOpReadVariableOpCmlp_ae_flattened_v0_1_hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ ё
+MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMulMatMul1MLP_Ae_Flattened_v0.1/dropout_1/Identity:output:0BMLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ к
;MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOpDmlp_ae_flattened_v0_1_hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: е
,MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAddBiasAdd5MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul:product:0CMLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
)MLP_Ae_Flattened_v0.1/hidden_layer_2/ReluRelu5MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Я
(MLP_Ae_Flattened_v0.1/dropout_2/IdentityIdentity7MLP_Ae_Flattened_v0.1/hidden_layer_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ и
8MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOpReadVariableOpAmlp_ae_flattened_v0_1_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: Џ
)MLP_Ae_Flattened_v0.1/output_layer/MatMulMatMul1MLP_Ae_Flattened_v0.1/dropout_2/Identity:output:0@MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ж
9MLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOpReadVariableOpBmlp_ae_flattened_v0_1_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:я
*MLP_Ae_Flattened_v0.1/output_layer/BiasAddBiasAdd3MLP_Ae_Flattened_v0.1/output_layer/MatMul:product:0AMLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0Ь
*MLP_Ae_Flattened_v0.1/output_layer/SoftmaxSoftmax3MLP_Ae_Flattened_v0.1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ё
IdentityIdentity4MLP_Ae_Flattened_v0.1/output_layer/Softmax:softmax:0<^MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp;^MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOp<^MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp;^MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOp9^MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOp8^MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOp:^MLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOp9^MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2x
:MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOp:MLP_Ae_Flattened_v0.1/hidden_layer_2/MatMul/ReadVariableOp2r
7MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOp7MLP_Ae_Flattened_v0.1/input_layer/MatMul/ReadVariableOp2t
8MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOp8MLP_Ae_Flattened_v0.1/input_layer/BiasAdd/ReadVariableOp2z
;MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp;MLP_Ae_Flattened_v0.1/hidden_layer_2/BiasAdd/ReadVariableOp2t
8MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOp8MLP_Ae_Flattened_v0.1/output_layer/MatMul/ReadVariableOp2x
:MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOp:MLP_Ae_Flattened_v0.1/hidden_layer_1/MatMul/ReadVariableOp2z
;MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp;MLP_Ae_Flattened_v0.1/hidden_layer_1/BiasAdd/ReadVariableOp2v
9MLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOp9MLP_Ae_Flattened_v0.1/output_layer/BiasAdd/ReadVariableOp: : : : : : : :1 -
+
_user_specified_nameinput_layer_input: 
Ж
a
C__inference_dropout_layer_call_and_return_conditional_losses_208036

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:€€€€€€€€€А*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
ї
c
*__inference_dropout_2_layer_call_fn_208147

inputs
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputs*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€ *
Tin
2*-
_gradient_op_typePartitionedCall-207699В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
÷	
г
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208110

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ †
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*'
_output_shapes
:€€€€€€€€€ *
T0Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
а
Ѓ
-__inference_output_layer_layer_call_fn_208170

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Е
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_208142

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ [

Identity_1IdentityIdentity:output:0*'
_output_shapes
:€€€€€€€€€ *
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :& "
 
_user_specified_nameinputs
е
∞
/__inference_hidden_layer_1_layer_call_fn_208064

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579*
Tout
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
©
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_207688

inputs
identityИQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *  А>C
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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€ М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ґ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Й
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€ o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€ i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:€€€€€€€€€ *
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :& "
 
_user_specified_nameinputs
Ж
a
C__inference_dropout_layer_call_and_return_conditional_losses_207551

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:€€€€€€€€€А*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
ђ

И
$__inference_signature_wrapper_207850
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-207839**
f%R#
!__inference__wrapped_model_207490*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€*
Tin
2	В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: : : 
й 
Э
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207819

inputs.
*input_layer_statefulpartitionedcall_args_1.
*input_layer_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИҐ&hidden_layer_1/StatefulPartitionedCallҐ&hidden_layer_2/StatefulPartitionedCallҐ#input_layer/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallФ
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputs*input_layer_statefulpartitionedcall_args_1*input_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207513*P
fKRI
G__inference_input_layer_layer_call_and_return_conditional_losses_207507*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:€€€€€€€€€А»
dropout/PartitionedCallPartitionedCall,input_layer/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-207563*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_207551*
Tout
2**
config_proto

GPU 

CPU2J 8є
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207585*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_207579ќ
dropout_1/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_207623*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€@*-
_gradient_op_typePartitionedCall-207635ї
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-207657*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_207651*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€ *
Tin
2ќ
dropout_2/PartitionedCallPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-207707*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_207695*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ ≥
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-207729*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_207723*
Tout
2Ф
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€А ::::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*ƒ
serving_default∞
P
input_layer_input;
#serving_default_input_layer_input:0€€€€€€€€€А @
output_layer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:на
ж,
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

regularization_losses
trainable_variables
	variables
	keras_api

signatures
t_default_save_signature
*u&call_and_return_all_conditional_losses
v__call__"Ћ)
_tf_keras_sequentialђ){"class_name": "Sequential", "name": "MLP_Ae_Flattened_v0.1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "MLP_Ae_Flattened_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 4096], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "MLP_Ae_Flattened_v0.1", "layers": [{"class_name": "Dense", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 4096], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
ї
regularization_losses
trainable_variables
	variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"ђ
_tf_keras_layerТ{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 4096], "config": {"batch_input_shape": [null, 4096], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
І

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"В
_tf_keras_layerи{"class_name": "Dense", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 4096], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 4096], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}}
Ђ
regularization_losses
trainable_variables
	variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ь
_tf_keras_layerВ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
А

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*}&call_and_return_all_conditional_losses
~__call__"џ
_tf_keras_layerЅ{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
±
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*&call_and_return_all_conditional_losses
А__call__"°
_tf_keras_layerЗ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
Б

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Џ
_tf_keras_layerј{"class_name": "Dense", "name": "hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
≤
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"°
_tf_keras_layerЗ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
€

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"Ў
_tf_keras_layerЊ{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
™
7iter
	8decay
9learning_rate
:momentum
;rho	rmsl	rmsm	rmsn	rmso	'rmsp	(rmsq	1rmsr	2rmss"
	optimizer
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
Ј
<non_trainable_variables
=layer_regularization_losses

regularization_losses
trainable_variables
	variables

>layers
?metrics
v__call__
t_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
Зserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
@non_trainable_variables
regularization_losses
Alayer_regularization_losses
trainable_variables
	variables

Blayers
Cmetrics
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
&:$
А А2input_layer/kernel
:А2input_layer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ
Dnon_trainable_variables
regularization_losses
Elayer_regularization_losses
trainable_variables
	variables

Flayers
Gmetrics
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
Ъ
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses
trainable_variables
	variables

Jlayers
Kmetrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
(:&	А@2hidden_layer_1/kernel
!:@2hidden_layer_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ
Lnon_trainable_variables
regularization_losses
Mlayer_regularization_losses
 trainable_variables
!	variables

Nlayers
Ometrics
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
Ы
Pnon_trainable_variables
#regularization_losses
Qlayer_regularization_losses
$trainable_variables
%	variables

Rlayers
Smetrics
А__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
':%@ 2hidden_layer_2/kernel
!: 2hidden_layer_2/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
Э
Tnon_trainable_variables
)regularization_losses
Ulayer_regularization_losses
*trainable_variables
+	variables

Vlayers
Wmetrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Xnon_trainable_variables
-regularization_losses
Ylayer_regularization_losses
.trainable_variables
/	variables

Zlayers
[metrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
%:# 2output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Э
\non_trainable_variables
3regularization_losses
]layer_regularization_losses
4trainable_variables
5	variables

^layers
_metrics
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
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
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
`0"
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
Ь
	atotal
	bcount
c
_fn_kwargs
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"е
_tf_keras_layerЋ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
Э
hnon_trainable_variables
dregularization_losses
ilayer_regularization_losses
etrainable_variables
f	variables

jlayers
kmetrics
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:.
А А2RMSprop/input_layer/kernel/rms
):'А2RMSprop/input_layer/bias/rms
2:0	А@2!RMSprop/hidden_layer_1/kernel/rms
+:)@2RMSprop/hidden_layer_1/bias/rms
1:/@ 2!RMSprop/hidden_layer_2/kernel/rms
+:) 2RMSprop/hidden_layer_2/bias/rms
/:- 2RMSprop/output_layer/kernel/rms
):'2RMSprop/output_layer/bias/rms
к2з
!__inference__wrapped_model_207490Ѕ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *1Ґ.
,К)
input_layer_input€€€€€€€€€А 
Т2П
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207741
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207967
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207932
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207762ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207993
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207796
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207831
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207980ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
с2о
G__inference_input_layer_layer_call_and_return_conditional_losses_208004Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_input_layer_layer_call_fn_208011Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ƒ2Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_208036
C__inference_dropout_layer_call_and_return_conditional_losses_208031і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
(__inference_dropout_layer_call_fn_208046
(__inference_dropout_layer_call_fn_208041і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ф2с
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208057Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_hidden_layer_1_layer_call_fn_208064Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_1_layer_call_and_return_conditional_losses_208089
E__inference_dropout_1_layer_call_and_return_conditional_losses_208084і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_1_layer_call_fn_208094
*__inference_dropout_1_layer_call_fn_208099і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ф2с
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208110Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_hidden_layer_2_layer_call_fn_208117Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_2_layer_call_and_return_conditional_losses_208142
E__inference_dropout_2_layer_call_and_return_conditional_losses_208137і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_2_layer_call_fn_208152
*__inference_dropout_2_layer_call_fn_208147і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
H__inference_output_layer_layer_call_and_return_conditional_losses_208163Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_output_layer_layer_call_fn_208170Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
=B;
$__inference_signature_wrapper_207850input_layer_input
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 Ђ
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_208057]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ћ
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207762v'(12CҐ@
9Ґ6
,К)
input_layer_input€€€€€€€€€А 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ £
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207831i'(12CҐ@
9Ґ6
,К)
input_layer_input€€€€€€€€€А 
p 

 
™ "К€€€€€€€€€А
-__inference_output_layer_layer_call_fn_208170O12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€£
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207796i'(12CҐ@
9Ґ6
,К)
input_layer_input€€€€€€€€€А 
p

 
™ "К€€€€€€€€€™
!__inference__wrapped_model_207490Д'(12;Ґ8
1Ґ.
,К)
input_layer_input€€€€€€€€€А 
™ ";™8
6
output_layer&К#
output_layer€€€€€€€€€•
C__inference_dropout_layer_call_and_return_conditional_losses_208031^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ •
E__inference_dropout_1_layer_call_and_return_conditional_losses_208084\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ •
E__inference_dropout_2_layer_call_and_return_conditional_losses_208137\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ •
E__inference_dropout_2_layer_call_and_return_conditional_losses_208142\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ }
(__inference_dropout_layer_call_fn_208041Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А•
C__inference_dropout_layer_call_and_return_conditional_losses_208036^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ •
E__inference_dropout_1_layer_call_and_return_conditional_losses_208089\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
(__inference_dropout_layer_call_fn_208046Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А}
*__inference_dropout_2_layer_call_fn_208147O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "К€€€€€€€€€ }
*__inference_dropout_2_layer_call_fn_208152O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "К€€€€€€€€€ ј
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207932k'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€А 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ®
H__inference_output_layer_layer_call_and_return_conditional_losses_208163\12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dropout_1_layer_call_fn_208094O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@¬
$__inference_signature_wrapper_207850Щ'(12PҐM
Ґ 
F™C
A
input_layer_input,К)
input_layer_input€€€€€€€€€А ";™8
6
output_layer&К#
output_layer€€€€€€€€€}
*__inference_dropout_1_layer_call_fn_208099O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Б
,__inference_input_layer_layer_call_fn_208011Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А 
™ "К€€€€€€€€€А©
G__inference_input_layer_layer_call_and_return_conditional_losses_208004^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ В
/__inference_hidden_layer_2_layer_call_fn_208117O'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ Ш
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207980^'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€А 
p

 
™ "К€€€€€€€€€Ћ
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207741v'(12CҐ@
9Ґ6
,К)
input_layer_input€€€€€€€€€А 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ј
Q__inference_MLP_Ae_Flattened_v0.1_layer_call_and_return_conditional_losses_207967k'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€А 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Г
/__inference_hidden_layer_1_layer_call_fn_208064P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@Ш
6__inference_MLP_Ae_Flattened_v0.1_layer_call_fn_207993^'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€А 
p 

 
™ "К€€€€€€€€€™
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_208110\'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ 