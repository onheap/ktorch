import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

plugins {
    java
    kotlin("jvm") version "1.8.10"
    application
}

group = "org.onheap"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:multik-core:0.2.1")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.1")
}

tasks.test {
    jvmArgs(listOf("--add-modules", "jdk.incubator.vector"))
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "17"
    kotlinOptions.freeCompilerArgs = listOf("-Xadd-modules=jdk.incubator.vector")
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<JavaExec> {
    jvmArgs(listOf("--add-modules", "jdk.incubator.vector"))
}

application {
    mainClass.set("MainKt")
}