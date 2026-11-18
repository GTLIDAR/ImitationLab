from imitationlab.data_tools.common.registry import (
    DatasetLoaderSpec,
    get_dataset_loader_spec,
    load_dataset_loader,
    register_dataset_loader,
    registered_dataset_loaders,
)


def test_registered_loaders_include_lafan1_aliases():
    names = registered_dataset_loaders()

    assert {"lafan1", "lafan1_csv"} <= set(names)


def test_load_lafan1_csv_loader():
    loader_cls = load_dataset_loader("lafan1-csv")

    assert loader_cls.__name__ == "Lafan1CsvLoader"


def test_register_dataset_loader_import_target():
    register_dataset_loader(
        "base_loader",
        module="imitationlab.data_tools.common.loader",
        class_name="BaseLoader",
    )

    spec = get_dataset_loader_spec("base-loader")
    assert spec == DatasetLoaderSpec(
        module="imitationlab.data_tools.common.loader",
        class_name="BaseLoader",
    )
    assert load_dataset_loader("base_loader").__name__ == "BaseLoader"
