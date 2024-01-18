from datasets import load_dataset, Dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")
data = [
    {"Smelly Sample":"\t@Override\n\tpublic void configSaved(RVConfigure configProg)\n\t{\n\t\tconfig.useBloom = bloomCB.isSelected();\n\t\tconfig.usePhong = phongCB.isSelected();\n\t\tconfig.useShadows = shadowCB.isSelected();\n\t\tconfig.useSoftShadows = softShadowCB.isSelected();\n\t\tconfig.useFsaa = fsaaCB.isSelected();\n\t\tconfig.useStereo = stereoCB.isSelected();\n\t\tconfig.useVsync = vsyncCB.isSelected();\n\n\t\ttry {\n\t\t\tconfig.fsaaSamples = Integer.parseInt(samplesTF.getText());\n\t\t} catch (Exception e) {\n\t\t\tsamplesTF.setText(config.fsaaSamples + \"\");\n\t\t}\n\n\t\ttry {\n\t\t\tconfig.shadowResolution = Integer.parseInt(shadowResTB.getText());\n\t\t} catch (Exception e) {\n\t\t\tshadowResTB.setText(config.shadowResolution + \"\");\n\t\t}\n\n\t\tconfig.targetFPS = (Integer) fpsSpinner.getValue();\n\t\tconfig.firstPersonFOV = (Integer) fpFovSpinner.getValue();\n\t\tconfig.thirdPersonFOV = (Integer) tpFovSpinner.getValue();\n\t\tconfig.frameX = (Integer) fxSpinner.getValue();\n\t\tconfig.frameY = (Integer) fySpinner.getValue();\n\t\tconfig.frameWidth = (Integer) fwSpinner.getValue();\n\t\tconfig.frameHeight = (Integer) fhSpinner.getValue();\n\t\tconfig.centerFrame = centerCB.isSelected();\n\t\tconfig.isMaximized = maximizedCB.isSelected();\n\t\tconfig.saveFrameState = saveStateCB.isSelected();\n\t}\n",
     "Method after Refactoring":"\t@Override\n\tpublic void configSaved(RVConfigure configProg)\n\t{\n\t\tconfig.useBloom = bloomCB.isSelected();\n\t\tconfig.usePhong = phongCB.isSelected();\n\t\tconfig.useShadows = shadowCB.isSelected();\n\t\tconfig.useSoftShadows = softShadowCB.isSelected();\n\t\tconfig.useFsaa = fsaaCB.isSelected();\n\t\tconfig.useStereo = stereoCB.isSelected();\n\t\tconfig.useVsync = vsyncCB.isSelected();\n\n\t\tupdateFsaaSamplesConfig(true);\n\t\tupdateShadowResolutionConfig(true);\n\n\t\tconfig.targetFPS = (Integer) fpsSpinner.getValue();\n\t\tconfig.firstPersonFOV = (Integer) fpFovSpinner.getValue();\n\t\tconfig.thirdPersonFOV = (Integer) tpFovSpinner.getValue();\n\t\tconfig.frameX = (Integer) fxSpinner.getValue();\n\t\tconfig.frameY = (Integer) fySpinner.getValue();\n\t\tconfig.frameWidth = (Integer) fwSpinner.getValue();\n\t\tconfig.frameHeight = (Integer) fhSpinner.getValue();\n\t\tconfig.centerFrame = centerCB.isSelected();\n\t\tconfig.isMaximized = maximizedCB.isSelected();\n\t\tconfig.saveFrameState = saveStateCB.isSelected();\n\t}\n",
     "Extracted Method":"\tprivate void updateFsaaSamplesConfig(boolean resetOnError)\n\t{\n\t\ttry {\n\t\t\tconfig.fsaaSamples = samplesTF.getInt();\n\t\t} catch (Exception e) {\n\t\t\tif (resetOnError) {\n\t\t\t\tsamplesTF.setText(config.fsaaSamples + \"\");\n\t\t\t}\n\t\t}\n\t}\n"},
    {"Smelly Sample":"        public String getResourceFileName(final int i) {\n            return String.format(\"META-INF/crest/%s/%s.%s.properties\", clazzName, commandName, i);\n        }\n",
     "Method after Refactoring":"    public String getResourceFileName(final int i) {\n        return getResourceFileName(clazzName, commandName, i);\n    }\n",
     "Extracted Method":"    public static String getResourceFileName(final String clazzName, final String commandName, final int i) {\n        return String.format(\"META-INF/crest/%s/%s.%s.properties\", clazzName, commandName, i);\n    }\n"}
]

formatted_dataset = {
    "source_method": [entry["Smelly Sample"] for entry in data],
    "refactored_method": [entry["Extracted Method"] for entry in data],
}
hf_dt = Dataset.from_dict(formatted_dataset)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=hf_dt,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

# print(type(dataset))
# print(dataset[0])