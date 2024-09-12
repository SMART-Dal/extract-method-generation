package org.refidentifier;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("Usage: java -jar refidentifier.jar <path-to-repo>");
            System.exit(1);
        }
        String repoPath = args[0];
        GetCommits getCommits = new GetCommits(repoPath);
        getCommits.getParentOfHead();
        System.out.println(new IdentifyRefactorings(getCommits.getRepository(), getCommits.getParentToLatestCommit(), getCommits.getLatestCommit())
                .identifyRefactoringInstances());
    }
}