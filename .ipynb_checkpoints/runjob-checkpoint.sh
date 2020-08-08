#runjob.sh
TRAINER_PACKAGE_PATH="/home/jupyter/titanic/train"
today="date +%Y_%m_%d_%H_%M_%S"
JOB_NAME="titanic_$today"
MAIN_TRAINER_MODULE="train.titanic_train"
JOB_DIR="gs://kubeflow-test-280609-kubeflowpipelines-default"
PACKAGE_STAGING_PATH="gs://kubeflow-test-280609-kubeflowpipelines-default/stage"
REGION="us-west1"
RUNTIME_VERSION="1.15"
PYTHON_VERSION="3.7"
WORK_BUCKET="gs://kubeflow-test-280609-kubeflowpipelines-default"
PREPROC_CSV_GCS_URI="$WORK_BUCKET/preprocdata/processed_train.csv"
MODEL_PKL_GCS_URI="$WORK_BUCKET/model/titanic_model.pkl"
ACC_CSV_GCS_URI="$WORK_BUCKET/latestacc/accuracy.csv"
gcloud ai-platform jobs submit training $JOB_NAME \
--scale-tier basic \
--package-path $TRAINER_PACKAGE_PATH \
--module-name $MAIN_TRAINER_MODULE \
--job-dir $JOB_DIR \
--region $REGION \
--staging-bucket $PACKAGE_STAGING_PATH \
--runtime-version $RUNTIME_VERSION \
--python-version $PYTHON_VERSION \
--\
--preproc_csv_gcs_uri $PREPROC_CSV_GCS_URI \
--model_pkl_gcs_uri $MODEL_PKL_GCS_URI \
--acc_csv_gcs_uri $ACC_CSV_GCS_URI \
--min_acc_progress 0.000001