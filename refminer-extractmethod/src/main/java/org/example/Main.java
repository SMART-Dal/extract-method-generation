package org.example;

public class Main {
    public static void main(String[] args) throws Exception {

        if (args.length < 3) {
            System.out.println("Please provide a repository URL, output file path and default branch as arguments.");
            System.exit(1);
        }
        String repoUrl = args[0];
        String outputFilePath = args[1];
        String defaultBranch = args[2];

        ExtractMethodExtractor extractMethodExtractor = new ExtractMethodExtractor(repoUrl,defaultBranch);
        extractMethodExtractor.identifyRefactoringInstances().generateSamples(outputFilePath);

    }
}