# this class define a user object that contains the user interests
class User:
    def __init__(self, topics=None, subtopics=None):
        self.topics = topics if topics is not None else []
        self.subtopics = subtopics if subtopics is not None else []

    def add_topic(self, topic):
        if topic not in self.topics:
            self.topics.append(topic)

    def add_subtopic(self, subtopic):
        if subtopic not in self.subtopics:
            self.subtopics.append(subtopic)

    def __repr__(self):
        return f"User(topics={self.topics}, subtopics={self.subtopics})"