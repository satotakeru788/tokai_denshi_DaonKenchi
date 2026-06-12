import { defineBackend } from '@aws-amplify/backend';
import { Duration, Stack } from 'aws-cdk-lib';
import * as cognito from 'aws-cdk-lib/aws-cognito';
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

// 開発者向け機能(モデル選択UIなど)の表示判定に使う Cognito グループ。
// フロントの src/userGroups.ts(DEVELOPERS_GROUP)と名前を一致させること。
//
// IMPORTANT: defineAuth({ groups: [...] }) は使わない。auth-construct はグループ
// ごとに「権限ゼロのIAMロール + RoleArn付きグループ」を自動生成し、IdentityPool
// に無条件設定済みの Token ロールマッピングによって所属ユーザーの一時クレデン
// シャルがその空ロールへ切り替わる — storage 権限も下の InvokeInferUrlPolicy も
// authenticated ロールにしか付いていないため、開発者だけ全機能が壊れる。
// (defineAuth({groups}) へ移行するなら、グループロールへのポリシー複製が必須。)
// RoleArn なしの素のグループなら IDトークンに cognito:groups が載るだけで、
// ロール解決は従来どおり authenticated ロールのまま。
// 配置は auth ネステッドスタック内 = userPoolId が同一スタック内参照になり、
// スタック間エッジを増やさない(auth を葉に保つ。下の循環コメント参照)。
const authStack = Stack.of(backend.auth.resources.userPool);
new cognito.CfnUserPoolGroup(authStack, 'DevelopersUserPoolGroup', {
  userPoolId: backend.auth.resources.userPool.userPoolId,
  groupName: 'developers',
  description: 'モデル選択UIなど開発者向け機能を表示するユーザー',
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
  // SnapStart restores a pre-initialized snapshot instead of cold-importing
  // numpy/scipy/onnxruntime (~4s) on every cold start — cuts cold latency.
  snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
});

// The function reads audio + models from the bucket and writes results back.
bucket.grantReadWrite(inferFn);

// SnapStart applies to PUBLISHED versions, and a Function URL can attach only to
// $LATEST or an alias — so expose the URL via an alias on the current version.
const liveAlias = new lambda.Alias(inferStack, 'LiveAlias', {
  aliasName: 'live',
  version: inferFn.currentVersion,
});

const inferUrl = liveAlias.addFunctionUrl({
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
      resources: [liveAlias.functionArn],
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
