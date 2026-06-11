import { defineBackend } from '@aws-amplify/backend';
import { Duration } from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { auth } from './auth/resource';
import { storage } from './storage/resource';

/**
 * DaonKenchi MVP backend: Cognito auth + S3 storage + a Python(ONNX) inference
 * Lambda exposed via a Function URL secured with AWS_IAM (SigV4) so only
 * logged-in users can call it.
 *
 * The Lambda is built WITHOUT Docker: deps come from a prebuilt manylinux
 * layer (tools/build_layer.py); models live in S3 (seeded by tools/seed_models.py),
 * not bundled in the function.
 */
const backend = defineBackend({
  auth,
  storage,
});

const inferStack = backend.createStack('infer');
const bucket = backend.storage.resources.bucket;

const depsLayer = new lambda.LayerVersion(inferStack, 'PyDepsLayer', {
  code: lambda.Code.fromAsset('amplify/layers/pydeps'),
  compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
  description: 'numpy / scipy / onnxruntime for DaonKenchi inference',
});

const inferFn = new lambda.Function(inferStack, 'InferFn', {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: 'handler.handler',
  code: lambda.Code.fromAsset('amplify/functions/infer'),
  layers: [depsLayer],
  memorySize: 2048,
  timeout: Duration.seconds(60),
  environment: {
    BUCKET_NAME: bucket.bucketName,
  },
  description: 'Estimate tire pressure (kPa) from a hammer-strike WAV stored in S3',
});

// The function reads audio + models from the bucket and writes results back.
bucket.grantReadWrite(inferFn);

const inferUrl = inferFn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.AWS_IAM,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [lambda.HttpMethod.POST],
    allowedHeaders: ['*'],
    maxAge: Duration.days(1),
  },
});

// Logged-in (Cognito authenticated) users may SigV4-invoke the Function URL.
// IMPORTANT: create the invoke policy IN the infer stack (attached to the auth
// role) rather than using inferUrl.grantInvokeUrl(authRole) — the latter adds an
// inline policy inside the AUTH stack referencing the infer stack, which closes a
// storage -> auth -> infer -> storage nested-stack cycle. Putting it here makes
// the edge infer -> auth, leaving auth a leaf (no cycle).
new iam.Policy(inferStack, 'InvokeInferUrlPolicy', {
  roles: [backend.auth.resources.authenticatedUserIamRole],
  statements: [
    new iam.PolicyStatement({
      actions: ['lambda:InvokeFunctionUrl'],
      resources: [inferFn.functionArn],
      conditions: { StringEquals: { 'lambda:FunctionUrlAuthType': 'AWS_IAM' } },
    }),
  ],
});

backend.addOutput({
  custom: {
    inferenceUrl: inferUrl.url,
    inferenceRegion: inferStack.region,
  },
});
