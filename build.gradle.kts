import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

plugins {
    java
    kotlin("jvm") version "1.8.10"
    application
    id("me.champeau.jmh") version "0.7.2"

    id("com.diffplug.spotless") version "6.23.3"
}

group = "org.onheap"

version = "1.0-SNAPSHOT"

repositories { mavenCentral() }

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:multik-core:0.2.2")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.2")
    implementation("org.ejml:ejml-all:0.43.1")
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

tasks.withType<JavaExec> { jvmArgs(listOf("--add-modules", "jdk.incubator.vector")) }

// == Benchmark configs start ==
jmh {
    warmupIterations.set(3) // Default is 10
    iterations.set(3) // Default is 10
    //    warmup.set("10s")      // Default is '10 s'
    //    timeOnIteration.set("10s")  // Default is '10 s'

    includes.addAll("benchmarks.NDArrayImplementationBenchmark")
}

tasks.jmhRunBytecodeGenerator { jvmArgs.addAll("--add-modules=jdk.incubator.vector") }

// == Benchmark configs end ==

spotless {
    java {
        importOrder()
        removeUnusedImports()

        // apply a specific flavor of google-java-format
        googleJavaFormat("1.19.1").aosp().reflowLongStrings().skipJavadocFormatting()
        // fix formatting of type annotations
        formatAnnotations()
    }

    kotlin {
        // by default the target is every '.kt' and '.kts` file in the java sourcesets
        ktfmt().dropboxStyle()
    }
    kotlinGradle {
        target("*.gradle.kts") // default target for kotlinGradle
        ktfmt().dropboxStyle()
    }
}

application { mainClass.set("MainKt") }
