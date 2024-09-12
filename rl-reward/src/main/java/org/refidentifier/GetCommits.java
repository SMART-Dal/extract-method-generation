package org.refidentifier;

import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.refactoringminer.api.GitService;
import org.refactoringminer.util.GitServiceImpl;

public class GetCommits {
    private final Repository repository;
    private String latestCommit;
    private String parentToLatestCommit;

    public GetCommits(String repoPath) throws Exception {
        GitService gitService = new GitServiceImpl();
        this.repository = gitService.openRepository(repoPath);
    }

    public void getParentOfHead() throws Exception {
        try (RevWalk revWalk = new RevWalk(this.repository)) {
            ObjectId lastCommitId = this.repository.resolve("HEAD");
            RevCommit commit = revWalk.parseCommit(lastCommitId);
            this.latestCommit = commit.getName();
            RevCommit parent = commit.getParent(0);
            this.parentToLatestCommit = parent.getName();
            revWalk.dispose();
        }
    }

    public String getLatestCommit() {
        return this.latestCommit;
    }

    public String getParentToLatestCommit() {
        return this.parentToLatestCommit;
    }

    public Repository getRepository() {
        return this.repository;
    }
}
