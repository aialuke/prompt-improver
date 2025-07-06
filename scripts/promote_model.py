import argparse

import mlflow


def promote_model(model_name, model_version, to_stage):
    """Promote a model version to a specified stage ('Staging' or 'Production')."""
    client = mlflow.tracking.MlflowClient()

    # Demote any existing model in the target stage to 'Archived'
    try:
        current_model = client.get_latest_versions(model_name, stages=[to_stage])[0]
        if current_model.version != model_version:
            print(
                f"Archiving version {current_model.version} currently in '{to_stage}'."
            )
            client.transition_model_version_stage(
                name=model_name, version=current_model.version, stage="Archived"
            )
    except IndexError:
        print(f"No model currently in '{to_stage}'.")

    # Promote the new model
    print(f"Promoting version {model_version} to '{to_stage}'.")
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=to_stage
    )
    print("Promotion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model to a new stage."
    )
    parser.add_argument(
        "--model-name", required=True, help="The name of the registered model."
    )
    parser.add_argument(
        "--model-version",
        required=True,
        type=int,
        help="The version of the model to promote.",
    )
    parser.add_argument(
        "--to-stage",
        required=True,
        choices=["Staging", "Production"],
        help="The target stage.",
    )

    args = parser.parse_args()

    promote_model(args.model_name, args.model_version, args.to_stage)
